#!/usr/bin/env python3

import os, csv, glob
from collections import defaultdict

try:
    import pydicom
except ImportError:
    pydicom = None


# Expected metadata headers from NBIA CSV
META_HEADERS = {
    "series_uid": "Series UID",
    "subject_id": "Subject ID",
    "num_images": "Number of Images",
    "file_location": "File Location",
    "modality": "Modality",
}

RAW_ROOT = "data/raw_dicom"
META_PATH = "data/metadata.csv"
INDEX_OUT = "data/index_summary.csv"
SELECT_OUT = "data/selected_series.csv"


def read_metadata(meta_path):
    """Read NBIA metadata.csv and extract key info if headers match."""
    rows = []
    if not os.path.exists(meta_path):
        return rows
    with open(meta_path, newline='') as f:
        rdr = csv.DictReader(f)
        if not all(h in rdr.fieldnames for h in META_HEADERS.values()):
            print("metadata.csv present but missing expected columns; falling back to filesystem scan.")
            return []
        for r in rdr:
            try:
                row = {
                    "PatientID": r[META_HEADERS["subject_id"]].strip(),
                    "SeriesInstanceUID": r[META_HEADERS["series_uid"]].strip(),
                    "SeriesPath": r[META_HEADERS["file_location"]].strip(),
                    "Modality": r.get(META_HEADERS["modality"], "").strip(),
                    "NumSlices": int(r[META_HEADERS["num_images"]]),
                }
                rows.append(row)
            except Exception:
                continue
    return rows


def normalize_series_path(raw_root, file_location):
    """
    Convert NBIA 'File Location' relative paths into real paths under data/raw_dicom.
    Example:
      ./LIDC-IDRI/LIDC-IDRI-0001/... -> data/raw_dicom/LIDC-IDRI-0001/...
    """
    if not file_location:
        return None
    loc = file_location.lstrip("./")
    if loc.startswith("LIDC-IDRI/"):
        candidate = os.path.join(raw_root, loc.split("/", 1)[1])
        if os.path.isdir(candidate):
            return candidate
    # fallback: search for directory tail
    tail = os.path.basename(os.path.normpath(loc))
    for d in glob.glob(os.path.join(raw_root, "**/"), recursive=True):
        if os.path.basename(os.path.normpath(d)) == tail and os.path.isdir(d):
            return d
    return None


def safe_dcm(path):
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return ds
    except Exception:
        return None


def scan_filesystem(raw_root):
    """Scan raw_dicom folders for CT series if metadata not available."""
    idx = []
    for pid in sorted(os.listdir(raw_root)):
        pdir = os.path.join(raw_root, pid)
        if not os.path.isdir(pdir):
            continue
        series_dirs = [d for d in glob.glob(os.path.join(pdir, "**/"), recursive=True)]
        series_dirs = [d for d in series_dirs if glob.glob(os.path.join(d, "*.dcm"))]
        for sdir in series_dirs:
            dcm_files = sorted(glob.glob(os.path.join(sdir, "*.dcm")))
            if not dcm_files:
                continue
            ds = safe_dcm(dcm_files[0])
            if ds is None:
                continue
            modality = getattr(ds, "Modality", "NA")
            suid = getattr(ds, "SeriesInstanceUID", os.path.abspath(sdir))
            idx.append({
                "PatientID": pid,
                "SeriesInstanceUID": suid,
                "SeriesPath": sdir,
                "Modality": modality,
                "NumSlices": len(dcm_files),
            })
    return idx


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(f"Expected raw DICOM directory not found: {RAW_ROOT}")

    # Try to read metadata first
    meta_rows = read_metadata(META_PATH)
    index_rows = []
    selected_rows = []

    if meta_rows:
        print("Using metadata.csv to index series.")
        per_patient_ct = defaultdict(list)
        for r in meta_rows:
            mapped = normalize_series_path(RAW_ROOT, r["SeriesPath"])
            row = {
                "PatientID": r["PatientID"],
                "SeriesInstanceUID": r["SeriesInstanceUID"],
                "SeriesPath": mapped if mapped else r["SeriesPath"],
                "Modality": r["Modality"] if r["Modality"] else "NA",
                "NumSlices": r["NumSlices"],
            }
            index_rows.append(row)
            if row["Modality"] == "CT" and mapped and os.path.isdir(mapped):
                per_patient_ct[row["PatientID"]].append(row)

        for pid, rows in per_patient_ct.items():
            best = max(rows, key=lambda t: t["NumSlices"])
            selected_rows.append(best)

        if not selected_rows:
            print("No valid CT series resolved from metadata; falling back to filesystem scan.")
            index_rows = scan_filesystem(RAW_ROOT)
    else:
        print("No usable metadata found; scanning filesystem.")
        index_rows = scan_filesystem(RAW_ROOT)

        per_patient = defaultdict(list)
        for r in index_rows:
            if r["Modality"] == "CT":
                per_patient[r["PatientID"]].append(r)
        for pid, rows in per_patient.items():
            best = max(rows, key=lambda t: t["NumSlices"])
            selected_rows.append(best)

    # Sort and write outputs
    index_rows = sorted(index_rows, key=lambda r: (r["PatientID"], r["Modality"], -int(r["NumSlices"])))
    selected_rows = sorted(selected_rows, key=lambda r: r["PatientID"])

    write_csv(INDEX_OUT, index_rows, ["PatientID","SeriesInstanceUID","SeriesPath","Modality","NumSlices"])
    write_csv(SELECT_OUT, selected_rows, ["PatientID","SeriesInstanceUID","SeriesPath","Modality","NumSlices"])

    print(f"Wrote {INDEX_OUT} (all series) and {SELECT_OUT} (one best CT series per patient).")


if __name__ == "__main__":
    main()
