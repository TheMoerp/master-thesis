#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from collections import Counter, defaultdict
from typing import Optional

import nibabel as nib


def find_subject_dirs(dataset_path: str):
    return [
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]


def find_volume_file(subject_dir: str, modality: Optional[str]):
    # Try requested modality first (e.g., t1c)
    if modality:
        hits = glob.glob(os.path.join(subject_dir, f"*-{modality}.nii.gz"))
        if hits:
            return hits[0]

    # Otherwise prefer common MR modalities
    for m in ["t1c", "t1", "t2", "flair"]:
        hits = glob.glob(os.path.join(subject_dir, f"*-{m}.nii.gz"))
        if hits:
            return hits[0]

    # Fallback: any NIfTI in subject directory
    any_hits = glob.glob(os.path.join(subject_dir, "*.nii.gz"))
    if any_hits:
        return any_hits[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Scan BraTS dataset and report unique volume shapes and voxel spacings."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/USERSPACE/hottgm5w/master-thesis/Brats/datasets/BraTS2025-GLI-PRE-Challenge-TrainingData",
        help="Path to BraTS dataset root (subjects as subfolders)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="t1c",
        help="Preferred modality to probe (e.g., t1c, t1, t2, flair). Uses fallbacks if missing.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    modality = args.modality

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)

    subject_dirs = find_subject_dirs(dataset_path)
    if not subject_dirs:
        print(f"No subject directories found under: {dataset_path}")
        sys.exit(1)

    shape_counter = Counter()
    spacing_counter = Counter()
    errors = []

    scanned_subjects = 0
    scanned_volumes = 0

    for subj in subject_dirs:
        fp = find_volume_file(subj, modality)
        if not fp:
            errors.append((subj, "no_nii_found"))
            continue
        try:
            img = nib.load(fp)
            shape = tuple(int(x) for x in img.shape[:3])
            zooms_full = img.header.get_zooms()
            zooms = tuple(float(x) for x in zooms_full[:3])

            shape_counter[shape] += 1
            spacing_counter[zooms] += 1

            scanned_subjects += 1
            scanned_volumes += 1
        except Exception as e:
            errors.append((subj, f"load_error: {e}"))

    # Summary
    print("BraTS volume size/spacing scan")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Preferred modality: {modality}")
    print(f"Subjects found: {len(subject_dirs)}")
    print(f"Subjects scanned: {scanned_subjects}")
    print(f"Volumes scanned:  {scanned_volumes}")
    print("-" * 60)

    if shape_counter:
        print("Unique shapes (count):")
        for shape, cnt in shape_counter.most_common():
            print(f"  {shape}: {cnt}")
    else:
        print("No shapes collected.")

    print("-" * 60)
    if spacing_counter:
        print("Unique voxel spacings (count):")
        for sp, cnt in spacing_counter.most_common():
            print(f"  {sp}: {cnt}")
    else:
        print("No spacings collected.")

    if errors:
        print("-" * 60)
        print(f"Warnings/Errors ({len(errors)}):")
        max_show = 10
        for subj, msg in errors[:max_show]:
            print(f"  {os.path.basename(subj)} -> {msg}")
        if len(errors) > max_show:
            print(f"  ... and {len(errors) - max_show} more")


if __name__ == "__main__":
    main()


