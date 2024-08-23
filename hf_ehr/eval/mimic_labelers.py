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


class Guo_LongLOSLabeler(Labeler):
    """Long LOS prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient stays in hospital for >=7 days.

    Excludes:
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.long_time: datetime.timedelta = datetime.timedelta(days=7)
        self.prediction_time_adjustment_func = move_datetime_to_end_of_day

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length >= `self.long_time`"""
        labels: List[Label] = []
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


class Guo_30DayReadmissionLabeler(TimeHorizonEventLabeler):
    """30-day readmissions prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of disharge whether the patient will be readmitted within 30 days.

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
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
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


class Guo_ICUAdmissionLabeler(WithinVisitLabeler):
    """ICU admission prediction task from Guo et al. 2023.

    Binary prediction task @ 11:59PM on the day of admission whether the patient will be admitted to the ICU during their admission.

    Excludes:
        - Patients transfered on same day as admission
        - Visits where discharge occurs on the same day as admission
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=move_datetime_to_end_of_day,
            visit_end_adjust_func=None,
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        # Return the start times of all ICU admissions -- this is our outcome
        return [e.start for e in get_icu_events(patient, self.ontology)]  # type: ignore

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return all inpatient visits where ICU transfer does not occur on the same day as admission."""
        # Get all inpatient visits -- each visit comprises a prediction (start, end) time horizon
        all_visits: List[Event] = get_inpatient_admission_events(patient, self.ontology)
        # Exclude visits where ICU admission occurs on the same day as admission
        icu_transfer_dates: List[datetime.datetime] = [
            x.replace(hour=0, minute=0, second=0, microsecond=0) for x in self.get_outcome_times(patient)
        ]
        valid_visits: List[Event] = []
        for visit in all_visits:
            # If admission and discharge are on the same day, then ignore
            if visit.start.date() == visit.end.date():
                continue
            # If ICU transfer occurs on the same day as admission, then ignore
            if visit.start.replace(hour=0, minute=0, second=0, microsecond=0) in icu_transfer_dates:
                continue
            valid_visits.append(visit)
        return valid_visits


##########################################################
##########################################################
# MIMIC-III Benchmark Tasks
# See: https://www.nature.com/articles/s41597-019-0103-9/figures/7 for
# details on how this was reproduced.
#
# Citation: Harutyunyan, H., Khachatrian, H., Kale, D.C. et al.
# Multitask learning and benchmarking with clinical time series data.
# Sci Data 6, 96 (2019). https://doi.org/10.1038/s41597-019-0103-9
##########################################################
##########################################################


class Harutyunyan_DecompensationLabeler(CodeLabeler):
    """Decompensation prediction task from Harutyunyan et al. 2019.

    Hourly binary prediction task on whether the patient dies in the next 24 hours.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        # Next 24 hours
        time_horizon = TimeHorizon(datetime.timedelta(hours=0), datetime.timedelta(hours=24))
        # Death events
        outcome_codes = list(get_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True))
        # Save ontology for `get_prediction_times()`
        self.ontology = ontology

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
        )

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of every hour after every ICU visit, up until death occurs or end of visit.
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        times: List[datetime.datetime] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        death_times: List[datetime.datetime] = self.get_outcome_times(patient)
        earliest_death_time: datetime.datetime = min(death_times) if len(death_times) > 0 else datetime.datetime.max
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, e.start, e.end, exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                # and only until the visit ends (`e.end`) or a death event occurs (`earliest_death_time`)
                end_of_stay: datetime.datetime = min(e.end, earliest_death_time)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < end_of_stay:
                    times.append(event_time)
                    event_time += datetime.timedelta(hours=1)
        return times


class Harutyunyan_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task from Harutyunyan et al. 2019.
    Single binary prediction task of whether patient dies within ICU admission 48 hours after admission.
    Make prediction 48 hours into ICU admission.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 48 hours
        - ICU admissions with no events before 48 hours
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        visit_start_adjust_func = lambda x: x + datetime.timedelta(
            hours=48
        )  # Make prediction 48 hours into ICU admission
        visit_end_adjust_func = lambda x: x
        super().__init__(ontology, visit_start_adjust_func, visit_end_adjust_func)

    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(get_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True))
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all ICU visits > 48 hours.

        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 48 hours
            - ICU admissions with no events before 48 hours
        """
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True)  # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        valid_events: List[Event] = []
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=48)
                and does_exist_event_within_time_range(
                    patient, e.start, e.start + datetime.timedelta(hours=48), exclude_event_idxs=icu_event_idxs
                )
            ):
                valid_events.append(e)
        return valid_events

if __name__ == '__main__':
    # TODO -- test labelers
    
    
    # TODO -- integrate these labelers with EHRSHOT
    