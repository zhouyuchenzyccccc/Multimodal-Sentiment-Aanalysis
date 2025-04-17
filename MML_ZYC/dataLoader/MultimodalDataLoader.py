import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib


class MultimodalDataLoader:
    def __init__(self, file_path, batch_size=20, test_size=0.2, val_size=0.1, random_state=42):
        self.file_path = file_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def _reshape_eye_pps(self, data):
        # Reshape eye (24,20,38) -> (480,38) 和 pps (24,20,230) -> (480,230)
        def strict_reshape(arr, target_dims):
            reshaped = arr.reshape(target_dims)
            if reshaped.shape != target_dims:
                raise ValueError(
                    f"Reshape失败: 原始形状{arr.shape} → 目标形状{target_dims}"
                )
            return reshaped

        return {
            'eeg': strict_reshape(data['features']['eeg'], (480, 32, 585)),
            'eye': strict_reshape(data['features']['eye'], (480, 38)),
            'pps': strict_reshape(data['features']['pps'], (480, 230)),
            'arousal': data['arousal_label'],
            'valence': data['valence_label']
        }



    def _subject_based_split(self, subject_ids):
        # 生成唯一被试列表并按比例划分
        unique_subjects = np.unique(subject_ids)
        subjects_train, subjects_temp = train_test_split(
            unique_subjects,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state
        )
        subjects_val, subjects_test = train_test_split(
            subjects_temp,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_state
        )
        return subjects_train, subjects_val, subjects_test

    def _get_dataset(self, data_dict):
        # 创建TensorDataset并进行标准化
        eeg = data_dict['eeg'].astype(np.float32)
        eye = data_dict['eye'].astype(np.float32)
        pps = data_dict['pps'].astype(np.float32)
        arousal = data_dict['arousal'].astype(np.int64) # 情感标签通常是整数
        valence = data_dict['valence'].astype(np.int64) # 情感标签通常是整数


        # 生成被试索引 (24个被试，每个20个样本)
        subject_ids = np.repeat(np.arange(24), 20)

        # 划分训练/验证/测试集
        train_subj, val_subj, test_subj = self._subject_based_split(subject_ids)

        # 生成样本mask
        train_mask = np.isin(subject_ids, train_subj)
        val_mask = np.isin(subject_ids, val_subj)
        test_mask = np.isin(subject_ids, test_subj)

        # 标准化处理（使用训练集统计量）
        eeg_train = eeg[train_mask]
        eeg_mean = eeg_train.mean(axis=(0, 2), keepdims=True)
        eeg_std = eeg_train.std(axis=(0, 2), keepdims=True)
        eeg = (eeg - eeg_mean) / eeg_std

        eye_train = eye[train_mask]
        eye_mean, eye_std = eye_train.mean(axis=0), eye_train.std(axis=0)
        eye = (eye - eye_mean) / eye_std

        pps_train = pps[train_mask]
        pps_mean, pps_std = pps_train.mean(axis=0), pps_train.std(axis=0)
        pps = (pps - pps_mean) / pps_std

        # 应用mask
        eeg_train, eeg_val, eeg_test = eeg[train_mask], eeg[val_mask], eeg[test_mask]
        eye_train, eye_val, eye_test = eye[train_mask], eye[val_mask], eye[test_mask]
        pps_train, pps_val, pps_test = pps[train_mask], pps[val_mask], pps[test_mask]
        arousal_train, arousal_val, arousal_test = arousal[train_mask], arousal[val_mask], arousal[test_mask]
        valence_train, valence_val, valence_test = valence[train_mask], valence[val_mask], valence[test_mask]


        # 转换为PyTorch张量
        train_set = TensorDataset(
            torch.from_numpy(eeg_train),
            torch.from_numpy(eye_train),
            torch.from_numpy(pps_train),
            torch.from_numpy(arousal_train),
            torch.from_numpy(valence_train)
        )

        val_set = TensorDataset(
            torch.from_numpy(eeg_val),
            torch.from_numpy(eye_val),
            torch.from_numpy(pps_val),
            torch.from_numpy(arousal_val),
            torch.from_numpy(valence_val)
        )

        test_set = TensorDataset(
            torch.from_numpy(eeg_test),
            torch.from_numpy(eye_test),
            torch.from_numpy(pps_test),
            torch.from_numpy(arousal_test),
            torch.from_numpy(valence_test)
        )


        return (
            (train_set, val_set, test_set),
            (eeg_mean, eeg_std, eye_mean, eye_std, pps_mean, pps_std)
        )

    def load_data(self):
        # 主加载函数
        raw_data = joblib.load(self.file_path)
        processed_data = self._reshape_eye_pps(raw_data)
        (train_set, val_set, test_set), stats = self._get_dataset(processed_data)

        # 创建DataLoader
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    data_loader = MultimodalDataLoader(file_path=r"F:\毕业设计\学长代码\Multimodal-sentiment-analysis\MML_ZYC\HCI_DATA\hci_data.pkl")
    train_loader, val_loader, test_loader = data_loader.load_data()
    # 调试：检查第一个batch的数据形状
    sample_batch = next(iter(train_loader))
    print("Sample batch shapes:")
    for i, data in enumerate(sample_batch):
        print(f"Data {i}: {data.shape if hasattr(data, 'shape') else len(data)}")
