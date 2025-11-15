#!/usr/bin/env python3
import os, csv, json, random

SELECT_CSV = "data/selected_series.csv"
OUT_JSON = "data/splits/split_by_patient.json"
SEED = 1337

def main():
    # collect patient IDs from selected_series.csv
    patients = []
    with open(SELECT_CSV, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pid = r["PatientID"].strip()
            if pid and pid not in patients:
                patients.append(pid)
    assert patients, "No patients found in selected_series.csv"

    random.seed(SEED)
    patients = sorted(patients)
    # deterministic shuffle for variety and also reproducibility
    random.shuffle(patients)

    # 20 patients -> 14/3/3 split
    n = len(patients)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    splits = {
        "train": patients[:n_train],
        "val":   patients[n_train:n_train+n_val],
        "test":  patients[n_train+n_val:]
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(splits, f, indent=2)
    print("Wrote", OUT_JSON, "->", splits)

if __name__ == "__main__":
    main()
