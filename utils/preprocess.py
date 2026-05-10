import os
import nibabel as nib
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data"
OUT_PATH = BASE_DIR / "data_v2"

IMG_SIZE = 256

TRAIN_IMG = OUT_PATH / "train" / "img"
TRAIN_MSK = OUT_PATH / "train" / "msk"
TEST_IMG = OUT_PATH / "test" / "img"
TEST_MSK = OUT_PATH / "test" / "msk"

for p in [TRAIN_IMG, TRAIN_MSK, TEST_IMG, TEST_MSK]:
    os.makedirs(p, exist_ok=True)

train_records = []
test_records = []
global_idx = 0


# =========================================================
# IMAGE NORMALIZATION
# =========================================================
def normalize(img):
    img = img.astype(np.float32)
    minv, maxv = np.min(img), np.max(img)

    if maxv - minv < 1e-8:
        return np.zeros_like(img)

    return (img - minv) / (maxv - minv)


# =========================================================
# RESIZE
# =========================================================
def resize_img(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def resize_msk(msk):
    return cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


# =========================================================
# MASK FIX (IMPORTANT)
# =========================================================
def fix_mask(msk):
    """
    Convert float mask → proper integer labels
    """
    msk = np.rint(msk)   # round values like 1.0, 2.0
    return msk.astype(np.uint8)


# =========================================================
def get_mask_path(img_path):
    return str(img_path).replace(".nii.gz", "_gt.nii.gz").replace(".nii", "_gt.nii")


# =========================================================
def save_sample(img, msk, patient, slice_id, split):
    global global_idx

    img_name = f"{split}_img_{global_idx:06d}.png"
    msk_name = f"{split}_msk_{global_idx:06d}.png"

    if split == "train":
        img_path = TRAIN_IMG / img_name
        msk_path = TRAIN_MSK / msk_name
        train_records.append([img_name, msk_name, patient, slice_id])
    else:
        img_path = TEST_IMG / img_name
        msk_path = TEST_MSK / msk_name
        test_records.append([img_name, msk_name, patient, slice_id])

    cv2.imwrite(str(img_path), img)

    # 👇 IMPORTANT FIX: preserve visibility without destroying labels
    msk_vis = (msk * 60).astype(np.uint8)
    cv2.imwrite(str(msk_path), msk_vis)

    global_idx += 1


# =========================================================
def process_patient(patient_path, split):

    files = sorted(os.listdir(patient_path))
    img_files = [f for f in files if "frame" in f and "gt" not in f]

    for img_file in img_files:

        if not img_file.endswith((".nii", ".nii.gz")):
            continue

        img_path = Path(patient_path) / img_file
        msk_path = Path(get_mask_path(img_path))

        if not msk_path.exists():
            continue

        try:
            img_nii = nib.load(str(img_path)).get_fdata()
            msk_nii = nib.load(str(msk_path)).get_fdata()
        except:
            continue

        if img_nii.shape != msk_nii.shape:
            continue

        depth = img_nii.shape[2]
        start, end = depth // 3, (2 * depth) // 3

        for i in range(start, end):

            img_slice = img_nii[:, :, i]
            msk_slice = msk_nii[:, :, i]

            if np.sum(msk_slice) == 0:
                continue

            # IMAGE
            img_slice = normalize(img_slice)
            img_slice = resize_img(img_slice)
            img_slice = (img_slice * 255).astype(np.uint8)

            # MASK (FIXED)
            msk_slice = fix_mask(msk_slice)
            msk_slice = resize_msk(msk_slice)

            save_sample(img_slice, msk_slice, patient_path.name, i, split)


# =========================================================
def run():

    train_root = RAW_PATH / "training"
    test_root = RAW_PATH / "testing"

    print("\n[INFO] TRAIN")
    for p in sorted(os.listdir(train_root)):
        path = train_root / p
        if path.is_dir():
            process_patient(path, "train")

    print("\n[INFO] TEST")
    for p in sorted(os.listdir(test_root)):
        path = test_root / p
        if path.is_dir():
            process_patient(path, "test")

    pd.DataFrame(train_records,
                 columns=["image", "mask", "patient", "slice"]
                 ).to_csv(OUT_PATH / "train.csv", index=False)

    pd.DataFrame(test_records,
                 columns=["image", "mask", "patient", "slice"]
                 ).to_csv(OUT_PATH / "test.csv", index=False)

    print("\n✔ DONE")
    print("Train:", len(train_records))
    print("Test:", len(test_records))


if __name__ == "__main__":
    run()