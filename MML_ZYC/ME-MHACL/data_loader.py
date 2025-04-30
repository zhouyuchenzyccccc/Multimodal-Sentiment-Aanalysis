# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Dataset for emotion recognition. Loads pre-saved NumPy arrays for EEG, eye-tracking, and physiological data.
    Assumes shapes: EEG (N,32,585), eye (N,38), physio (N,230), labels (N,2).
    """

    def __init__(self, eeg_path, eye_path, phy_path, label_path, transform=None):
        self.eeg_data = np.load(eeg_path)  # shape (N,32,585)
        self.eye_data = np.load(eye_path)  # shape (N,38)
        self.phy_data = np.load(phy_path)  # shape (N,230)
        self.labels = np.load(label_path)  # shape (N,2), each label is binary (0/1)
        assert len(self.eeg_data) == len(self.eye_data) == len(self.phy_data) == len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load modalities
        eeg = self.eeg_data[idx].astype(np.float32)  # (32,585)
        eye = self.eye_data[idx].astype(np.float32)  # (38,)
        phy = self.phy_data[idx].astype(np.float32)  # (230,)
        label = self.labels[idx].astype(np.int64)  # e.g., [arousal, valence]
        # Optionally apply transforms (e.g., data augmentation)
        if self.transform:
            eeg, eye, phy = self.transform(eeg, eye, phy)
        # Convert to torch Tensors
        eeg = torch.from_numpy(eeg)  # (32,585)
        eye = torch.from_numpy(eye)  # (38,)
        phy = torch.from_numpy(phy)  # (230,)
        return eeg, eye, phy, label


def default_augment(eeg, eye, phy, noise_eeg=0.01, noise_eye=0.05, noise_phy=0.05):
    """
    Basic augmentation: add Gaussian noise to each modality.
    The parameters control noise magnitude.
    """
    eeg_aug = eeg + np.random.normal(0, noise_eeg, eeg.shape).astype(np.float32)
    eye_aug = eye + np.random.normal(0, noise_eye, eye.shape).astype(np.float32)
    phy_aug = phy + np.random.normal(0, noise_phy, phy.shape).astype(np.float32)
    return eeg_aug, eye_aug, phy_aug


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive pre-training. Wraps EmotionDataset to produce two augmented views of each sample.
    """

    def __init__(self, base_dataset, augment=default_augment):
        self.base = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        eeg, eye, phy, _ = self.base[idx]  # ignore labels
        # First view
        eeg1, eye1, phy1 = self.augment(eeg, eye, phy)
        # Second view
        eeg2, eye2, phy2 = self.augment(eeg, eye, phy)
        # Convert to tensors
        eeg1 = torch.from_numpy(eeg1);
        eye1 = torch.from_numpy(eye1);
        phy1 = torch.from_numpy(phy1)
        eeg2 = torch.from_numpy(eeg2);
        eye2 = torch.from_numpy(eye2);
        phy2 = torch.from_numpy(phy2)
        return eeg1, eye1, phy1, eeg2, eye2, phy2
