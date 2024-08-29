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

    def get_outcome_times(self, patient: Patient, selected_discharge_time: datetime.datetime) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions that occur after the selected discharge time."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            if admission_time > selected_discharge_time:  # Ensure only subsequent admissions are considered
                times.append(admission_time)
        return times

    def get_prediction_times(self, selected_discharge_time: datetime.datetime) -> List[datetime.datetime]:
        """Return the prediction time set to 11:59 PM on the day of the selected discharge."""
        prediction_time = self.prediction_time_adjustment_func(selected_discharge_time)
        return [prediction_time]

    def label(self, patient: Patient, selected_admission_time: datetime.datetime, selected_discharge_time: datetime.datetime) -> List[Label]:
        """Label the selected admission with readmission status."""
        labels: List[Label] = []
        outcome_times = self.get_outcome_times(patient, selected_discharge_time)

        # Get the prediction time for the selected discharge
        prediction_times = self.get_prediction_times(selected_discharge_time)
        prediction_time = prediction_times[0]

        # Check if any of the subsequent admissions occur on the same day as the selected discharge
        same_day_admission = any(
            outcome_time.replace(hour=0, minute=0, second=0, microsecond=0) == prediction_time.replace(hour=0, minute=0, second=0, microsecond=0)
            for outcome_time in outcome_times
        )
        if same_day_admission:
            return labels  # Skip labeling for this discharge if readmitted on the same day

        # Determine if the patient was readmitted within 30 days after the selected discharge
        is_readmitted = any(
            selected_discharge_time < outcome_time < selected_discharge_time + self.time_horizon.end
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

    # Process each split
    for split_name in splits:
        dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split=split_name)
        
        # Get the patient IDs
        patient_ids = dataset.get_pids()
        
        # Add tqdm progress bar for processing patients
        for pid in tqdm(patient_ids, desc=f"Processing split: {split_name}"):
            patient = dataset.femr_db[pid]

            # Get all inpatient admissions for the patient
            admission_times = get_inpatient_admission_discharge_times(patient, ontology)
            
            # Filter out admissions where the patient was less than 18 years old or where admission and discharge were on the same day
            snomed_event_time = next((event.start for event in patient.events if event.code == "SNOMED/3950001"), None)
            birthdate = snomed_event_time

            filtered_admissions = [
                (admission_time, discharge_time)
                for admission_time, discharge_time in admission_times
                if birthdate is not None and calculate_age(birthdate, admission_time) >= 18 and admission_time.date() != discharge_time.date()
            ]
            
            if not filtered_admissions:
                continue
            
            # Randomly select one admission from the filtered list
            selected_admission_time, selected_discharge_time = random.choice(filtered_admissions)

            # Apply each labeler to the selected admission
            for labeler_name, labeler in labelers.items():
                labels = labeler.label(patient, selected_admission_time, selected_discharge_time)
                
                # Count total and positive labels
                total_labeler_stats[labeler_name]["total_labels"] += len(labels)
                total_labeler_stats[labeler_name]["positive_labels"] += sum(label.value for label in labels)

    # Print final results in a table format
    print(f"| {'Task':<20} | {'# of Total Labels':<20} | {'# of Positive Labels':<20} |")
    print(f"|{'-'*22}|{'-'*22}|{'-'*22}|")
    for labeler_name, stats in total_labeler_stats.items():
        print(f"| {labeler_name:<20} | {stats['total_labels']:<20} | {stats['positive_labels']:<20} |")