# Import necessary libraries and modules
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import femr.datasets
from typing import Dict, List, Tuple, Set, Union, Optional, Callable
from torch.utils.data import Dataset
from hf_ehr.config import Event, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
from hf_ehr.data.tokenization import DescTokenizer
from femr import Patient
from femr.extension import datasets as extension_datasets
from femr.labelers.core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler
from hf_ehr.data.datasets import FEMRDataset
from femr.labelers.omop import (
    CodeLabeler,
    WithinVisitLabeler,
    get_death_concepts,
    get_inpatient_admission_events,
    move_datetime_to_end_of_day,
)
import datetime
from collections import deque, defaultdict
#from femr.labelers.omop_inpatient_admissions import get_inpatient_admission_discharge_times
import time
import random
from datetime import timedelta
import csv

# Define a fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# TODO -- @Suhana
##########################################################
##########################################################
# CLMBR Benchmark Tasks
# See: https://www.medrxiv.org/content/10.1101/2022.04.15.22273900v1
# details on how this was reproduced.
#
# Citation: Guo et al.
# "EHR foundation models improve robustness in the presence of temporal distribution shift"
# Scientific Reports. 2023.
##########################################################
##########################################################

def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP", "Visit/ERIP"]

def get_inpatient_admission_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return get_femr_codes(
        ontology, get_inpatient_admission_concepts(), is_ontology_expansion=False, is_silent_not_found_error=True
    )

def get_inpatient_admission_events(patient: Patient, ontology: extension_datasets.Ontology) -> List[Event]:
    admission_codes: Set[str] = get_inpatient_admission_codes(ontology)
    events: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes and e.omop_table == "visit_occurrence":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            events.append(e)
    return events


def get_inpatient_admission_discharge_times(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    events: List[Event] = get_inpatient_admission_events(patient, ontology)
    times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for e in events:
        if e.end is None:
            raise RuntimeError(f"Event {e} cannot have `None` as its `end` attribute.")
        if e.start > e.end:
            raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
        times.append((e.start, e.end))
    return times

class InstantLabValueLabeler(Labeler):
    """Apply a multi-class label for the outcome of a lab test.

    Prediction Time: Immediately before lab result is returned (i.e. 1 minute before)
    Time Horizon: The next immediate result for this lab test
    Label: Severity level of lab

    Excludes:
        - Labels that occur at the same exact time as the very first event in a patient's history
    """

    # parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology
        self.outcome_codes: Set[str] = get_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )

    def label(self, patient: Patient, is_show_warnings: bool = False) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            if patient.events[0].start == e.start:
                # Ignore events that occur at the same time as the first event in the patient's history
                continue
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    try:
                        # `e.unit` is string of form "mg/dL", "ounces", etc.
                        label: int = self.label_to_int(self.value_to_label(str(e.value), str(e.unit)))
                        prediction_time: datetime.datetime = e.start - datetime.timedelta(minutes=1)
                        labels.append(Label(prediction_time, label))
                    except Exception as exception:
                        if is_show_warnings:
                            print(
                                f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                                f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                                f" | Exception: {exception}"
                            )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "categorical"

    def label_to_int(self, label: str) -> int:
        if label == "normal":
            return 0
        elif label == "mild":
            return 1
        elif label == "moderate":
            return 2
        elif label == "severe":
            return 3
        raise ValueError(f"Invalid label without a corresponding int: {label}")

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"


# TODO - check in with Michael - some patients were missing labels bc they didn't have VISIT/IP in their timeline
class LongLOSLabeler(Labeler):
    """Long LOS prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days.
    Here, we first look for patients with an inpatient visit (i.e. they have the code "VISIT/IP", then we get their 
    admission and discharge times, calculate the length of stay and assign it a label of true if it is >= 7 days)

    Excludes:
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.long_time: datetime.timedelta = datetime.timedelta(days=7)
        # move prediction time to the end of the day
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day
    
    def label(self, patient: Patient, admission_time: datetime.datetime, discharge_time: datetime.datetime) -> List[Label]:
        """Label the selected admission with admission length >= `self.long_time`."""
        labels: List[Label] = []
        
        # Check if the selected admission is considered long (>= 7 days)
        is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
        prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)

        # Apply the label to the selected admission
        labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"

class Readmission30DayLabeler(TimeHorizonEventLabeler):
    """30-day readmissions prediction task.

    Binary prediction task @ 11:59PM on the day of discharge whether the patient will be readmitted within 30 days.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology
        self.time_horizon = TimeHorizon(
            start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
        )
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return prediction times set to 11:59 PM on the day of discharge."""
        prediction_times: List[datetime.datetime] = []
        for _, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time = self.prediction_time_adjustment_func(discharge_time)
            prediction_times.append(prediction_time)
        return prediction_times

    def label(self, patient: Patient, selected_admission_time: datetime.datetime, selected_discharge_time: datetime.datetime) -> List[Label]:
        """Label the selected admission with readmission status."""
        labels: List[Label] = []
        outcome_times = self.get_outcome_times(patient)

        # Set the prediction time to 11:59 PM on the day of discharge
        prediction_time: datetime.datetime = self.prediction_time_adjustment_func(selected_discharge_time)
        
        # Ignore patients who are readmitted on the same day they were discharged (to prevent data leakage)
        admission_times = set(
            outcome_time.replace(hour=0, minute=0, second=0, microsecond=0) for outcome_time in outcome_times
        )
        if prediction_time.replace(hour=0, minute=0, second=0, microsecond=0) in admission_times:
            return labels  # Skip labeling for this discharge if readmitted on the same day

        # Determine if the patient was readmitted within 30 days after the selected discharge
        is_readmitted = any(
            selected_discharge_time < outcome_time <= selected_discharge_time + self.time_horizon.end
            for outcome_time in outcome_times
        )

        # Add the label for this admission
        labels.append(Label(prediction_time, is_readmitted))
        return labels

    def get_time_horizon(self) -> TimeHorizon:
        """Return the time horizon for the 30-day readmission task."""
        return self.time_horizon

class MortalityLabeler(Labeler):
    """In-hospital mortality prediction task.

    Binary prediction task @ 11:59PM on the day of admission whether the patient dies during their hospital stay.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient, selected_admission_time: datetime.datetime, selected_discharge_time: datetime.datetime) -> List[Label]:
        """Label the selected admission with mortality if death occurs during the stay."""
        labels: List[Label] = []
        outcome_times = self.get_outcome_times(patient)

        # Check if any outcome (death) occurs before or at the discharge time
        is_mortality = any(outcome_time <= selected_discharge_time for outcome_time in outcome_times)
        prediction_time: datetime.datetime = self.prediction_time_adjustment_func(selected_admission_time)
        labels.append(Label(prediction_time, is_mortality))
        return labels

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome (i.e., death)."""
        outcome_codes = list(get_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_labeler_type(self) -> LabelType:
        """Return the type of labeler (boolean in this case)."""
        return "boolean"


def _get_all_children(ontology: extension_datasets.Ontology, code: str) -> Set[str]:
    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) > 0:
        temp_parent_code: str = parent_deque.popleft()
        try:
            for temp_child_code in ontology.get_children(temp_parent_code):
                children_code_set.add(temp_child_code)
                parent_deque.append(temp_child_code)
        except:
            pass

    return children_code_set

def get_femr_codes(
    ontology: extension_datasets.Ontology,
    omop_concept_codes: List[str],
    is_ontology_expansion: bool = True,
    is_silent_not_found_error: bool = True,
) -> Set[str]:
    """Does ontology expansion on the given OMOP concept codes if `is_ontology_expansion` is True."""
    if not isinstance(omop_concept_codes, list):
        omop_concept_codes = [omop_concept_codes]
    codes: Set[str] = set()
    for omop_concept_code in omop_concept_codes:
        try:
            expanded_codes = (
                _get_all_children(ontology, omop_concept_code) if is_ontology_expansion else {omop_concept_code}
            )
            codes.update(expanded_codes)
        except ValueError:
            if not is_silent_not_found_error:
                raise ValueError(f"OMOP Concept Code {omop_concept_code} not found in ontology.")
    return codes


class ThrombocytopeniaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
        "LOINC/777-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"


class HyperkalemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
        "LOINC/6298-4",
        "LOINC/2823-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mmol/l"):
                # mmol/L
                # Original OMOP concept ID: 8753
                value = value
            elif unit.startswith("meq/l"):
                # mEq/L (1-to-1 -> mmol/L)
                # Original OMOP concept ID: 9557
                value = value
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 18 to get mmol/L)
                # Original OMOP concept ID: 8840
                value = value / 18.0
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value > 7:
            return "severe"
        elif value > 6.0:
            return "moderate"
        elif value > 5.5:
            return "mild"
        return "normal"


class HypoglycemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
        "LOINC/14749-6",
        # "LOINC/15074-8",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mg/dl"):
                # mg / dL
                # Original OMOP concept ID: 8840, 9028
                value = value / 18
            elif unit.startswith("mmol/l"):
                # mmol / L (x 18 to get mg/dl)
                # Original OMOP concept ID: 8753
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 3:
            return "severe"
        elif value < 3.5:
            return "moderate"
        elif value <= 3.9:
            return "mild"
        return "normal"


class HyponatremiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_codes = ["LOINC/LG11363-5", "LOINC/2951-2", "LOINC/2947-0"]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 125:
            return "severe"
        elif value < 130:
            return "moderate"
        elif value <= 135:
            return "mild"
        return "normal"


class AnemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for anemia based on hemoglobin levels (g/L).
    Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""

    original_omop_concept_codes = [
        "LOINC/LP392452-1",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("g/dl"):
                # g / dL
                # Original OMOP concept ID: 8713
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value * 10
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 1000 to get g/dL)
                # Original OMOP concept ID: 8840
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value / 100
            elif unit.startswith("g/l"):
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 70:
            return "severe"
        elif value < 110:
            return "moderate"
        elif value < 120:
            return "mild"
        return "normal"
""""
def print_specific_and_inpatient_event(patient: Patient) -> None:
    #Print the event with code SNOMED/3950001 and the first inpatient admission event with its code.
    specific_event = None
    first_inpatient_event = None

    # Use the get_inpatient_admission_discharge_times function to find inpatient admission events
    inpatient_admissions = get_inpatient_admission_discharge_times(patient, ontology)
    
    # Check if there are inpatient admissions
    if inpatient_admissions:
        first_inpatient_event = inpatient_admissions[0]  # Get the first inpatient admission

        # Iterate over events to find the specific event
        for event in patient.events:
            if event.code == "SNOMED/3950001" and specific_event is None:
                specific_event = event
                break

        print(f"Patient ID: {patient.patient_id}")

        if specific_event:
            print(f"  Specific event (SNOMED/3950001):")
            print(f"    Event start: {specific_event.start}")
            print(f"    Event value: {specific_event.value}")
            print(f"    Event unit: {specific_event.unit}")
            print(f"    Event OMOP table: {specific_event.omop_table}")
        else:
            print("  No event with code SNOMED/3950001 found.")

        # Iterate over events to find the code for the first inpatient admission
        first_admission_code = None
        for event in patient.events:
            if event.start == first_inpatient_event[0] and event.end == first_inpatient_event[1]:
                first_admission_code = event.code
                break

        admission_time, discharge_time = first_inpatient_event
        print(f"  First inpatient admission event:")
        print(f"    Admission time: {admission_time}")
        print(f"    Discharge time: {discharge_time}")
        print(f"    Admission code: {first_admission_code}")
        print("------------")
"""

def calculate_age(birthdate: datetime.datetime, event_time: datetime.datetime) -> int:
    return event_time.year - birthdate.year

# Function to process a dataset split
def process_split(dataset, labelers, ontology):
    labeler_stats = {}

    patient_ids = dataset.get_pids()

    for pid in tqdm(patient_ids, desc=f"Processing patients in split"):
        patient = dataset.femr_db[pid]

        # Get all inpatient admissions for the patient
        admission_times = get_inpatient_admission_discharge_times(patient, ontology)
        
        # Filter out admissions where the patient was less than 18 years old or where admission and discharge were on the same day
        snomed_event_time = next((event.start for event in patient.events if event.code == "SNOMED/3950001"), None)
        birthdate = snomed_event_time

        # Filter out admissions where the patient was less than 18 years old or where admission and discharge were on the same day
        filtered_admissions = [
            (admission_time, discharge_time)
            for admission_time, discharge_time in admission_times
            if birthdate is not None and calculate_age(birthdate, admission_time) >= 18 and admission_time.date() != discharge_time.date()
        ]
        
        if not filtered_admissions:
            continue
        
        # Randomly select one admission from the filtered list
        selected_admission_time, selected_discharge_time = random.choice(filtered_admissions)

        # Apply all labelers to the selected admission
        for labeler_name, labeler in labelers.items():
            if labeler_name not in labeler_stats:
                labeler_stats[labeler_name] = {
                    "total_labels": 0,
                    "positive_labels": 0
                }

            labels = labeler.label(patient, selected_admission_time, selected_discharge_time)

            if labels:
                labeler_stats[labeler_name]["total_labels"] += len(labels)
                for label in labels:
                    if label.value:  # Assuming positive labels are those with a true/positive value
                        labeler_stats[labeler_name]["positive_labels"] += 1

    return labeler_stats

if __name__ == '__main__':
    from hf_ehr.config import PATH_TO_FEMR_EXTRACT_MIMIC4  # Ensure correct path

    # Load the FEMR dataset
    splits = ['train', 'val', 'test']
    
    # Load the ontology
    ontology = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split='train').femr_db.get_ontology()

    # Initialize the labelers
    labelers = {
        "LongLOS": LongLOSLabeler(ontology=ontology),
        "Readmission30Day": Readmission30DayLabeler(ontology=ontology),
        "Mortality": MortalityLabeler(ontology=ontology),
    }

    # Initialize counters for total results across all splits
    total_labeler_stats = {labeler_name: {"total_labels": 0, "positive_labels": 0} for labeler_name in labelers.keys()}

    # Limit to a few patients for testing
    num_patients_to_test = 5  # You can increase this number for full dataset processing

    # Process each split
    for split_name in splits:
        dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split=split_name)
        
        # Get a limited number of patient IDs for testing
        patient_ids = dataset.get_pids()
        
        # Process the split for all labelers
        labeler_stats = process_split(dataset, labelers, ontology)

        for labeler_name, stats in labeler_stats.items():
            total_labeler_stats[labeler_name]["total_labels"] += stats["total_labels"]
            total_labeler_stats[labeler_name]["positive_labels"] += stats["positive_labels"]

    # Print final results in a table format
    print(f"| {'Task':<20} | {'# of Total Labels':<20} | {'# of Positive Labels':<20} |")
    print(f"|{'-'*22}|{'-'*22}|{'-'*22}|")
    for labeler_name, stats in total_labeler_stats.items():
        print(f"| {labeler_name:<20} | {stats['total_labels']:<20} | {stats['positive_labels']:<20} |")

    # Save the results to a CSV file
    csv_filename = "labeler_stats_all_splits.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        fieldnames = ['Task', 'Total Labels', 'Positive Labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for labeler_name, stats in total_labeler_stats.items():
            writer.writerow({
                'Task': labeler_name,
                'Total Labels': stats['total_labels'],
                'Positive Labels': stats['positive_labels']
            })

    print(f"Results have been saved to {csv_filename}")