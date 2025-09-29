#!/usr/bin/env python
"""
Custom pipeline proposed: 
  1. Brain extraction (HD-BET)
  2. N4 bias-field correction (SimpleITK)
  3. Copy into nnU-Net folder layout with correct naming
"""

import subprocess, shutil, json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nib
import SimpleITK as sitk                 
from tqdm import tqdm


RAW_ROOT   = Path("C:/Users/super/Desktop/DATASETS/ATLAS_2/Training")
TEST_ROOT  = Path("C:/Users/super/Desktop/DATASETS/ATLAS_2/Testing")   
NNUNET_RAW = Path("./nnunet_raw/Task201_ATLAS2")

OUT_IMG   = NNUNET_RAW / "imagesTr"
OUT_LAB   = NNUNET_RAW / "labelsTr"
OUT_TEST  = NNUNET_RAW / "imagesTs"                                     

for d in (OUT_IMG, OUT_LAB, OUT_TEST):
    d.mkdir(parents=True, exist_ok=True)


def find_cases(root: Path):
    for t1 in root.glob("R*/sub-*/ses-1/anat/*_T1w.nii.gz"):
        cand = list(t1.parent.glob("*lesion_mask.nii.gz"))             #ensure mask exists  
        mask = cand[0]
        cid = t1.stem.split("_ses-")[0]  
        yield t1, mask, cid

def find_test_cases(root: Path):                                        
    for t1 in root.glob("R*/sub-*/ses-1/anat/*_T1w.nii.gz"):
        # subject_id
        cid = t1.stem.split("_ses-")[0]#
        yield t1, cid

 

def hd_bet(src, dst):
    subprocess.run(
        ["hd-bet", "-i", src, "-o", dst, "-device", "cuda"],
        check=True, stdout=subprocess.PIPE
    )

def n4_bias_correct(src, brain_mask, dst):
    img  = sitk.ReadImage(str(src), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(brain_mask), sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    sitk.WriteImage(corrector.Execute(img, mask), str(dst))

 

def save_to_nnunet(final_img, lesion_mask, case_id):
    # save image with _0000 suffix for nnUNet
    nib.save(final_img, OUT_IMG / f"{case_id}_0000.nii.gz")
    # save label without _0000 suffix
    shutil.copy2(lesion_mask, OUT_LAB / f"{case_id}.nii.gz")

def save_test_img(final_img, case_id):                                 
    # save test image with _0000 suffix for nnUNet
    nib.save(final_img, OUT_TEST / f"{case_id}_0000.nii.gz")

def main():
 
    cases = list(find_cases(RAW_ROOT))
    for t1, mask, cid in tqdm(cases, desc="train cases"):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bet_path = tmp / "brain.nii.gz"
            hd_bet(t1, bet_path)

            n4_path = tmp / "n4.nii.gz"
            n4_bias_correct(bet_path, bet_path, n4_path)

            save_to_nnunet(nib.load(n4_path), mask, cid)

    # test set
    test_cases = list(find_test_cases(TEST_ROOT))
    for t1, cid in tqdm(test_cases, desc="test cases"):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bet_path = tmp / "brain.nii.gz"
            hd_bet(t1, bet_path)

            n4_path = tmp / "n4.nii.gz"
            n4_bias_correct(bet_path, bet_path, n4_path)

            save_test_img(nib.load(n4_path), cid)

    # include metadata
    dataset_json = {
        "name": "ATLAS2_T1w",
        "description": "Stroke lesion segmentation on T1-weighted MRIs",
        "tensorImageSize": "3D",
        "reference": "Liew et al., Sci Data 2022",
        "licence": "CC BY-NC-SA",
        "release": "0.1",
        "modality": {"0": "T1"},
        "labels": {"0": "background", "1": "lesion"},
        "numTraining": len(cases),
        "numTest": len(test_cases),
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz",
             "label": f"./labelsTr/{cid}.nii.gz"} for _, _, cid in cases
        ],
        "test": [f"./imagesTs/{cid}_0000.nii.gz" for _, cid in test_cases]
    }
    (NNUNET_RAW / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

if __name__ == "__main__":
    main() 