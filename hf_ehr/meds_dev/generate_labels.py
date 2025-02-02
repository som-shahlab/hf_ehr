"""
TAKEN FROM: https://github.com/som-shahlab/mimic_tutorial/blob/53ed809c9ce3b4d6c36cf91e4f40224887acb769/generate_labels.py#L4
CREDIT ETHAN STEINBERG
"""

import femr.labelers
import meds_reader
import meds
import datetime
from tqdm import tqdm

import os
import shutil
import pyarrow as pa
import pyarrow.csv as pacsv

from typing import List, Mapping, Iterator, Optional, Tuple
import pandas as pd
import config
import itertools
import functools

############################################
# 
# Taken from FEMR 2.4
# See here: https://github.com/som-shahlab/femr/blob/47d687a943d60c04647aa05d35e33a086d09ee10/src/femr/labelers/core.py#L27
#
############################################
def _label_map_func(subjects: Iterator[meds_reader.Subject], *, labeler) -> List[meds.Label]:
    result = []
    for subject in tqdm(subjects):
        result.extend(labeler.label(subject))
    return result

class Labeler():
    
    def apply(
        self,
        db: meds_reader.SubjectDatabase,
    ) -> List[meds.Label]:
        """Apply the `label()` function one-by-one to each Subject in a sequence of Subjects.

        Args:
            dataset (datasets.Dataset): A HuggingFace Dataset with meds_reader.Subject objects to be labeled.
            num_proc (int, optional): Number of CPU threads to parallelize across. Defaults to 1.

        Returns:
            A list of labels
        """
        return list(itertools.chain.from_iterable(db.map(functools.partial(_label_map_func, labeler=self))))

############################################
# End of FEMR 2.4 code
############################################

class MIMICInpatientMortalityLabeler(Labeler):
    def __init__(self, time_after_admission: datetime.timedelta):
        self.time_after_admission = time_after_admission

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        # Get admission and discharge times
        death_times = set()
        admission_starts = dict()
        admission_ends = dict()
        for event in subject.events:
            if event.code.startswith('HOSPITAL_ADMISSION'):
                admission_starts[event.hadm_id] = event.time
            if event.code.startswith('HOSPITAL_DISCHARGE'):
                admission_ends[event.hadm_id] = event.time
            if event.code == meds.death_code:
                death_times.add(event.time)

        assert len(death_times) in (0, 1)
        assert admission_starts.keys() == admission_ends.keys(), f'{subject} {admission_starts.keys()} {admission_ends.keys()}'
        admission_ranges = {(admission_starts[k], admission_ends[k]) for k in admission_starts.keys()}

        if len(death_times) == 1:
            death_time = list(death_times)[0]
        else:
            death_time = datetime.datetime(9999, 1, 1) # Very far in the future
            
        
        labels = []
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            if prediction_time >= death_time:
                continue

            is_death = death_time < admission_end
            labels.append(meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=is_death))

        return labels

class MIMICLongAdmissionLabeler(Labeler):
    def __init__(self, time_after_admission: datetime.timedelta, admission_length: datetime.timedelta):
        self.time_after_admission = time_after_admission
        self.admission_length = admission_length

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        admission_starts = dict()
        admission_ends = dict()

        for event in subject.events:
            if event.code.startswith('HOSPITAL_ADMISSION'):
                admission_starts[event.hadm_id] = event.time
            if event.code.startswith('HOSPITAL_DISCHARGE'):
                admission_ends[event.hadm_id] = event.time

        assert admission_starts.keys() == admission_ends.keys(), f'{subject} {admission_starts.keys()} {admission_ends.keys()}'

        admission_ranges = {(admission_starts[k], admission_ends[k]) for k in admission_starts.keys()}

        labels = []
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            is_long_admission = (admission_end - admission_start) > self.admission_length

            labels.append(meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=is_long_admission))
        
        return labels

labelers: Mapping[str, Labeler] = {
    'death': MIMICInpatientMortalityLabeler(time_after_admission=datetime.timedelta(hours=48)),
    'long_los': MIMICLongAdmissionLabeler(time_after_admission=datetime.timedelta(hours=48), admission_length=datetime.timedelta(days=7)),
}

def main():
    if os.path.exists('labels'):
        shutil.rmtree('labels')
    os.mkdir('labels')

    with meds_reader.SubjectDatabase(config.database_path, num_threads=6) as database:
        for label_name in config.label_names:
            labeler = labelers[label_name]
            # for key in database:
            #     labeler.label(database[key])
            labels = labeler.apply(database)
            print(f'Generated {len(labels)} labels for {label_name}')

            label_frame = pa.Table.from_pylist(labels, meds.label_schema)
            pacsv.write_csv(label_frame, os.path.join('labels', label_name + '.csv'))

if __name__ == "__main__":
    main()