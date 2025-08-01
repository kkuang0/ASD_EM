import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Dict, Optional, Callable


class EMAxonDataset(Dataset):
    """Dataset for EM axon cross-section images with multi-task labels."""

    def __init__(self,
                 df: pd.DataFrame,
                 transform: Optional[Callable] = None,
                 task_mapping: Optional[Dict] = None):
        """Create dataset from dataframe.

        Args:
            df: DataFrame with columns ['filepath', 'pathology', 'region', 'depth']
            transform: Image transformations
            task_mapping: Custom label mapping for tasks
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Default task mapping
        if task_mapping is None:
            self.task_mapping = {
                'pathology': {'Control': 0, 'ASD': 1},
                'region': {'A25': 0, 'A46': 1, 'OFC': 2},
                'depth': {'DWM': 0, 'SWM': 1},
            }
        else:
            self.task_mapping = task_mapping

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        labels = {}
        for task, mapping in self.task_mapping.items():
            if row[task] in mapping:
                labels[task] = mapping[row[task]]
            else:
                labels[task] = 0
        return img, labels

    def get_class_distribution(self) -> Dict[str, Dict]:
        distribution = {}
        for task in self.task_mapping.keys():
            if task in self.df.columns:
                distribution[task] = self.df[task].value_counts().to_dict()
        return distribution


class LabeledEMDataset(EMAxonDataset):
    """Example dataset returning the standard label mapping."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = {
            'pathology': 0 if row['pathology'] == 'Control' else 1,
            'region': {'A25': 0, 'A46': 1, 'OFC': 2}[row['region']],
            'depth': 0 if row['depth'] == 'DWM' else 1,
        }
        return img, labels


class PathDepthPairDataset(Dataset):
    """Dataset that combines pathology and depth into a single label."""

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_mapping = {
            ('Control', 'DWM'): 0,
            ('Control', 'SWM'): 1,
            ('ASD', 'DWM'): 2,
            ('ASD', 'SWM'): 3,
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_mapping.get((row['pathology'], row['depth']), 0)
        return img, {'pair': label}

