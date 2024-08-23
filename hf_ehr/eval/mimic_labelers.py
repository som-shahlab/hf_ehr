# Import necessary libraries and modules
import os
from tqdm import tqdm
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
from collections import deque
from femr.labelers.omop_inpatient_admissions import get_inpatient_admission_discharge_times
import time

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

def get_icu_events(
    patient: Patient, ontology: extension_datasets.Ontology, is_return_idx: bool = False
) -> Union[List[Event], List[Tuple[int, Event]]]:
    """Return all ICU events for this patient.
    If `is_return_idx` is True, then return a list of tuples (event, idx) where `idx` is the index of the event in `patient.events`.
    """
    icu_visit_detail_codes: Set[int] = get_icu_visit_detail_codes(ontology)
    events: Union[List[Event], List[Tuple[int, Event]]] = []
    for idx, e in enumerate(patient.events):
        # `visit_detail` is more accurate + comprehensive than `visit_occurrence` for ICU events for STARR OMOP for some reason
        if e.code in icu_visit_detail_codes and e.omop_table == "visit_detail":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(
                    f"Event {e} for patient {patient.patient_id} cannot have `None` as its `start` or `end` attribute."
                )
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} for patient {patient.patient_id} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            if is_return_idx:
                events.append((idx, e))  # type: ignore
            else:
                events.append(e)
    return events

def get_icu_visit_detail_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return get_femr_codes(
        ontology, get_icu_visit_detail_concepts(), is_ontology_expansion=True, is_silent_not_found_error=True
    )


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
        # move prediction time to the end of the day -> get_inpatient_admission_events -> get_inpatient_admission_codes -> get_inpatient_admission_concepts ("Visit/IP")
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length >= `self.long_time`"""
        labels: List[Label] = []
        # get_inpatient_admission_discharge_times -> 
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            # If admission and discharge are on the same day, then ignore
            if admission_time.date() == discharge_time.date():
                continue
            is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class Readmission30DayLabeler(TimeHorizonEventLabeler):
    """30-day readmissions prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of disharge whether the patient will be readmitted within 30 days.
    The prediction time is moved to the end of the discharge day

    Excludes:
        - Patients readmitted on same day as discharge
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(minutes=1), end=datetime.timedelta(days=30)
        )
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions that occur after the initial discharge."""
        times: List[datetime.datetime] = []
        admissions = get_inpatient_admission_discharge_times(patient, self.ontology)
        
        for i in range(1, len(admissions)):  # Start from the second admission
            admission_time, discharge_time = admissions[i]
            previous_discharge_time = admissions[i-1][1]  # Discharge time of the previous admission

            # Ensure we only consider admissions that occur after the previous discharge
            if admission_time > previous_discharge_time:
                times.append(admission_time)

        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction timm."""
        times: List[datetime.datetime] = []
        admission_times = set()
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)
            # Ignore patients who are readmitted the same day they were discharged b/c of data leakage
            if prediction_time.replace(hour=0, minute=0, second=0, microsecond=0) in admission_times:
                continue
            times.append(prediction_time)
            admission_times.add(admission_time.replace(hour=0, minute=0, second=0, microsecond=0))
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


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


class InstantLabValueLabeler(Labeler):
    """Apply a multi-class label for the outcome of a lab test.

    Prediction Time: Immediately before lab result is returned (i.e. 1 minute before)
    Time Horizon: The next immediate result for this lab test
    Label: Severity level of lab

    Excludes:
        - Labels that occur at the same exact time as the very first event in a patient's history
    """

    # Parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(self, ontology: extension_datasets.Ontology):
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
                continue
            if e.code in self.outcome_codes:
                if e.value is not None:
                    try:
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
        print(f"Raw value: {raw_value}, Unit: {unit}")
        try:
            value = float(raw_value)
            if unit and unit.startswith('mmol/L'):
                if value < 3.0:
                    return "severe"
                else:
                    return "normal"
            # Add other unit handling if necessary
        except ValueError:
            return "normal"  # Default to normal if value conversion fails


class HypoglycemiaLabValueLabeler(InstantLabValueLabeler):
    """Labeler for hypoglycemia based on blood glucose lab values."""
    original_omop_concept_codes = ["SNOMED/33747003", "LOINC/LP416145-3"]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert blood glucose value to a binary hypoglycemia label based on mg/dL."""
        value = float(raw_value)

        if value < 70:  # Hypoglycemia threshold in mg/dL
            return "severe"  # This will be interpreted as True in the binary context
        else:
            return "normal"  # This will be interpreted as False in the binary context
    
    def label_to_int(self, label: str) -> int:
        if label == "severe":
            return 1  # Indicate hypoglycemia
        else:
            return 0  # Indicate no hypoglycemia

class OMOPConceptCodeLabeler(CodeLabeler):
    """Same as CodeLabeler, but add the extra step of mapping OMOP concept IDs
    (stored in `omop_concept_ids`) to femr codes (stored in `codes`)."""

    # parent OMOP concept codes, from which all the outcome
    # are derived (as children from our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        outcome_codes: List[int] = list(
            get_femr_codes(
                ontology,
                self.original_omop_concept_codes,
                is_ontology_expansion=True,
            )
        )
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func
            if prediction_time_adjustment_func
            else identity,
        )


class AnemiaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at least one explicitly coded occurrence of Anemia."""

    # LOINC concept codes related to anemia
    original_omop_concept_codes = [
        'LOINC/LP392452-1',  # Add additional relevant LOINC codes if needed
        'LOINC/718-7'        # Hemoglobin concentration
    ]

    # Threshold for anemia diagnosis (e.g., 13.0 g/dL for men, 12.0 g/dL for women, so we decide on 13.0)
    anemia_threshold = 13.0

    def value_to_label(self, raw_value: Optional[str], unit: Optional[str]) -> str:
        """Convert to a binary anemia label. Anemia is identified by the value being below the threshold."""
        print(f"Debug: Raw value: {raw_value}, Unit: {unit}")
        if unit == 'g/dL':
            try:
                value = float(raw_value)
                if value < self.anemia_threshold:
                    return "anemia"  # Label as anemia if below the threshold
            except ValueError:
                pass  # Handle cases where conversion to float fails
        return "no_anemia"  # Default to no anemia if not below the threshold or incorrect unit

    def label_to_int(self, label: str) -> int:
        """Convert label to binary value."""
        print(f"Debug: Converting label '{label}' to binary.")
        if label == "anemia":
            return 1  # Indicate anemia
        return 0  # Indicate no anemia

    def __init__(self, ontology: extension_datasets.Ontology):
        self.ontology = ontology
        self.outcome_codes: Set[str] = get_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )
        print(f"AnemiaCodeLabeler initialized with outcome codes: {self.outcome_codes}")

    def label(self, patient: Patient, is_show_warnings: bool = False) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            # Skip the first event (or apply other filtering criteria if needed)
            if patient.events[0].start == e.start:
                continue
            # Check if the event code matches one of the outcome codes
            if e.code in self.outcome_codes:
                print(f"Event code: {e.code}, Value: {e.value}, Unit: {e.unit}, Time: {e.start}")
                try:
                    # Use value_to_label to get the appropriate label
                    label_value: str = self.value_to_label(e.value, e.unit)
                    label: int = self.label_to_int(label_value)
                    # Adjust prediction time to be one minute before the event start time
                    prediction_time: datetime.datetime = e.start - datetime.timedelta(minutes=1)
                    print(f"Debug: Prediction time is {prediction_time}, Label: {label}")
                    labels.append(Label(prediction_time, label))
                except Exception as exception:
                    if is_show_warnings:
                        print(
                            f"Warning: Error processing code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                            f" | Exception: {exception}"
                        )
        return labels


if __name__ == '__main__':
    from hf_ehr.config import PATH_TO_FEMR_EXTRACT_MIMIC4  # Ensure correct path
    print("Testing HypoglycemiaLabeler on patients in FEMRDataset (MIMIC4):")

    # Load the FEMR dataset
    train_dataset = FEMRDataset(PATH_TO_FEMR_EXTRACT_MIMIC4, split='train')

    # Load the ontology
    ontology = train_dataset.femr_db.get_ontology()

    # Initialize the anemia labeler
    anemia_labeler = AnemiaCodeLabeler(ontology=ontology)

    # Counter to limit to 10 patients
    patient_counter = 0
    max_patients = 50

    for pid in train_dataset.get_pids():
        if patient_counter >= max_patients:
            break
        
        patient = train_dataset.femr_db[pid]
        labels = anemia_labeler.label(patient)

        if any(label.value for label in labels):
            print(f"\nPatient ID: {pid}")
            for label in labels:
                if label.value:
                    print(f"  Prediction time: {label.time}, Anemia: {label.value}")

        patient_counter += 1


    # TODO -- test labelers
    
    
    # TODO -- integrate these labelers with EHRSHOT
    