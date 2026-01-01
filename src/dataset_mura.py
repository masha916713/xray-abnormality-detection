from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

@dataclass
class Sample:
    img_path: str
    label: int  # 0/1

def _read_labeled_studies(csv_path: str) -> List[Tuple[str, int]]:
    """
    CSV format (typical MURA):
      relative_study_path,label
    Example:
      MURA-v1.1/train/XR_WRIST/patient00001/study1_positive,1
    """
    studies = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            study_rel = parts[0].strip()
            label = int(parts[1].strip())
            studies.append((study_rel, label))
    return studies

def build_image_samples(mura_root: str, labeled_csv: str) -> List[Sample]:
    """
    Expand each study into its image files (usually PNGs).
    Each image inherits the study label.
    """
    csv_path = os.path.join(mura_root, labeled_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    studies = _read_labeled_studies(csv_path)
    samples: List[Sample] = []

    for study_rel, y in studies:
        # study_rel might already include "MURA-v1.1/..." or be relative.
        # We'll handle both:
        if study_rel.startswith("MURA"):
            study_abs = os.path.join(os.path.dirname(mura_root), study_rel)
        else:
            study_abs = os.path.join(mura_root, study_rel)

        # images are commonly .png in MURA
        imgs = sorted(glob.glob(os.path.join(study_abs, "*.png")))
        if not imgs:
            # sometimes nested views; try recursive
            imgs = sorted(glob.glob(os.path.join(study_abs, "**", "*.png"), recursive=True))

        for img_path in imgs:
            samples.append(Sample(img_path=img_path, label=y))

    if not samples:
        raise RuntimeError("No image samples found. Check dataset path and CSV format.")
    return samples

class MuraImageDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor([float(s.label)], dtype=torch.float32)  # shape [1]
        return img, y, s.img_path
