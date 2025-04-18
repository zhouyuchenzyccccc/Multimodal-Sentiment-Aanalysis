# @usage: 封装成Torch.Dataset

import os
import copy
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader


from data.LoadFeatures import DataFeatures
from common import utils


class FeatureDataset(Dataset):
    def __init__(
            self,
            features: DataFeatures,
            ex_nums=48,
            mode="train",
            test_person=-1,
            cls_num=2,
            dependent=False,
            n_splits=5,
    ):
        """
        Introduction:
            FeatureDataset类，用于封装数据集
        Args:
            features: DataFeatures类
            ex_num: 每个受试者的样本数量
            mode: 数据集模式，train/test
            test_person: 测试受试者id / 第几折
            cls_num: 类别数量
            dependent: 是否为跨被试实验
            n_splits: 交叉验证折数
        """

        self.features = features.features.copy()
        self.labels = features.label
        self.mode = mode
        self.cls_num = cls_num
        self.ex_nums = ex_nums
        self.indices = np.arange(len(self.labels))


        # 二分类标签处理
        if cls_num == 2:
            self.indices = self.filter_binary_labels()

        if dependent:
            self.split_data_dependent(mode, n_splits, current_split=test_person)
        else:
            self.split_data_independent(mode, test_person)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = {modality: feature[idx] for modality, feature in self.features.items()}
        return data, self.labels[idx]

    def filter_binary_labels(self):
        """
        筛选二分类标签 (仅保留标签0和2，并将2映射为1)
        """
        indices = np.where((self.labels == 0) | (self.labels == 2))[0]
        # self.labels = self.labels[indices]
        self.labels = np.where(self.labels == 2, 1, self.labels)
        # for modality in self.features.keys():
        #     self.features[modality] = self.features[modality][indices]
        return indices

    def split_data(self, indices):
        """通用的数据划分逻辑，用于独立和跨被试实验"""
        for modality, feature in self.features.items():
            self.features[modality] = feature[indices]
        self.labels = self.labels[indices]

    def split_data_independent(self, mode, test_person):
        """
        Introduction:
            独立被试实验，根据mode和test_person划分数据集
        Args:
            mode: 数据集模式，train/valid/test
            test_person: 测试受试者id
        """
        assert test_person < (
                len(self.labels) // self.ex_nums
        ), "测试受试者id超出已有受试者id范围"

        start, end = (
            test_person * self.ex_nums,
            (test_person + 1) * self.ex_nums,
        )
        indices = self.indices

        # 找到属于测试者范围内的二分类数据索引
        test_indices = indices[(indices >= start) & (indices < end)]
        train_indices = np.setdiff1d(indices, test_indices)

        if mode == "train":
            self.split_data(train_indices)
        elif mode == "test":
            self.split_data(test_indices)
        else:
            raise ValueError("mode should be 'train' or 'test'")

    def split_data_dependent(self, mode, n_splits, current_split):
        """
        跨被试实验，根据mode、n_splits和current_split划分数据集

        Args:
            mode: 数据集模式，train/test
            n_splits: 交叉验证折数
            current_split: 当前使用第几折作为测试集
        """
        if current_split < 0 or current_split >= n_splits:
            raise ValueError("current_split must be in the range [0, n_splits)")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 获取当前折对应的索引
        for i, (train_indices, test_indices) in enumerate(kf.split(self.indices)):
            if i == current_split:
                if mode == "train":
                    self.split_data(self.indices[train_indices])
                elif mode == "test":
                    self.split_data(self.indices[test_indices])
                else:
                    raise ValueError("mode should be 'train' or 'test'")
                break


def load_data(config, test_person=-1):
    """加载数据集"""
    label_type = config["data"]["HCI"]["label_type"]
    data = DataFeatures(
        data_path=config["data"]["HCI"]["data_path"],
        modalities=config["training"]["using_modalities"],
        subject_lists=config["data"]["HCI"]["subject_lists"],
        Norm="Z_score",
        label_type=label_type,
    )
    train_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["HCI"]["ex_nums"],
        mode="train",
        test_person=test_person,
        cls_num=config["num_classes"],
        dependent=config["training"]["dependent"],
        n_splits=config["training"]["n_folds"],
    )
    test_dataset = FeatureDataset(
        data,
        ex_nums=config["data"]["HCI"]["ex_nums"],
        mode="test",
        test_person=test_person,
        cls_num=config["num_classes"],
        dependent=config["training"]["dependent"],
        n_splits=config["training"]["n_folds"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["HCI"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["HCI"]["num_workers"],
    )
    return train_loader, test_loader




