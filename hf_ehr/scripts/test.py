
import collections
import femr.datasets

femr_db = femr.datasets.PatientDatabase(path_to_femr_db)

# Collect lab values
numericals = collections.defaultdict(list)
for p_idx, patient in enumerate(femr_db):
    path_to_output = f"/share/pi/nigam/mwornow/hf_ehr/cache/create_cookbook/numericals/numericals_{p_idx // 100_000}.json"
    if os.path.exists(path_to_output):
        continue
    weight = 1/(len(femr_db) * len(patient.events))
    for e in patient.events:
        if (
            e.value is not None # `value` is not None
            and ( # `value` is numeric
                isinstance(e.value, float)
                or isinstance(e.value, int)
            )
        ):
            # Numeric
            numericals[e.code].append((e.value, weight))
    if (p_idx + 1) % 100_000 == 0:
        print(f"Processed {p_idx + 1} patients")
        # Save to file
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
        with open(path_to_output, "w") as f:
            json.dump(numericals, f)
        numericals = collections.defaultdict(list)

# Bin lab values
n_bins: int = 10
numerical_bins = collections.defaultdict(list)
for n_idx, code, values in enumerate(numericals.items()):
    path_to_output = f"/share/pi/nigam/mwornow/hf_ehr/cache/create_cookbook/numerical_bins_{n_bins}/numerical_bins_{n_idx // 1_000}.json"
    quantiles = np.percentile([x[0] for x in values ], np.arange(0, 101, n_bins))
    total_weight = np.sum([ x[1] for x in values ])
    weight_per_bucket = total_weight / n_bins
    numerical_bins[code] = {
        'quantiles': quantiles,
        'weight' : [ weight_per_bucket for _ in range(n_bins) ],
    }
    if (n_idx + 1) % 1_000 == 0:
        print(f"Processed {n_idx + 1} codes")
        # Save to file
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
        with open(path_to_output, "w") as f:
            json.dump(numerical_bins, f)
        numerical_bins = collections.defaultdict(list)


    # Do individual weighting per bin
    # bucket_indices = np.digitize(data, bins=quantiles)
    # buckets = [ np.where(bucket_indices == i)[0] for i in range(0, len(quantiles) + 1) ]
    # weights = [ np.sum(values[x]) for x in buckets ]
    # numerical_bins[code] = {
    #     'quantiles': quantiles,
    #     'weight' : [ x * np.log(x) + (1 - x) * np.log(1 - x) for x in weights ],
    # }