import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from data.LoadFeatures import DataFeatures


class CrossSubjectDataLoader:
    def __init__(self, file_path, batch_size=32, random_state=42):
        self.file_path = file_path
        self.batch_size = batch_size

        self.random_state = random_state

    def _reshape_eye_pps(self, data):
        # Reshape eye (24,20,38) -> (480,38) 和 pps (24,20,230) -> (480,230)
        data_path = r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl"
        subject_list = [
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            10,
            11,
            13,
            14,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
        ]
        print(subject_list)
        modalities = ["eeg", "eye", "pps"]
        mahnobData = DataFeatures(
            data_path,
            modalities=modalities,
            subject_lists=subject_list,
            Norm="Z_score",
            label_type="arousal",
        )
        eeg = mahnobData.features["eeg"]
        eye = mahnobData.features["eye"]
        pps = mahnobData.features["pps"]

        # # 检查NaN和Inf
        # for name, arr in [('eeg', eeg), ('eye', eye), ('pps', pps)]:
        #     if np.isnan(arr).any():
        #         print(f"警告: {name} 包含NaN值")
        #     if np.isinf(arr).any():
        #         print(f"警告: {name} 包含Inf值")

        return {
            'eeg': eeg,
            'eye': eye,
            'pps': pps,
            'arousal': data['arousal_label'],
            'valence': data['valence_label']
        }

    def _subject_based_split(self, subject_ids):
        """
        不再随机划分被试，而是固定每个被试的训练/测试样本数量。
        返回每个被试的训练和测试样本的索引。
        """
        train_indices = []
        test_indices = []
        val_indices = []  # 添加验证集索引

        train_samples = 16
        val_samples = 1


        for subject_id in np.unique(subject_ids):
            # 获取当前被试的所有样本的索引
            subject_indices = np.where(subject_ids == subject_id)[0]

            # 随机打乱当前被试的样本索引
            np.random.seed(self.random_state)  # 保证可重复性
            np.random.shuffle(subject_indices)

            # 划分训练集、验证集和测试集
            train_indices.extend(subject_indices[:train_samples])
            val_indices.extend(subject_indices[train_samples:train_samples + val_samples])  # 划分验证集
            test_indices.extend(subject_indices[train_samples + val_samples:])  # 划分测试集

        return train_indices, val_indices, test_indices

    def _get_dataset(self, data_dict):
        # 创建TensorDataset并进行标准化
        eeg = data_dict['eeg'].astype(np.float32)
        eye = data_dict['eye'].astype(np.float32)
        pps = data_dict['pps'].astype(np.float32)
        arousal = data_dict['arousal'].astype(np.int64)  # 情感标签通常是整数
        valence = data_dict['valence'].astype(np.int64)  # 情感标签通常是整数

        # 生成被试索引 (24个被试，每个20个样本)
        subject_ids = np.repeat(np.arange(24), 20)
        print(subject_ids)

        # 划分训练/验证/测试集
        train_indices, val_indices, test_indices = self._subject_based_split(subject_ids)

        # 应用索引
        eeg_train = eeg[train_indices]
        eye_train = eye[train_indices]
        pps_train = pps[train_indices]
        arousal_train = arousal[train_indices]
        valence_train = valence[train_indices]

        eeg_val = eeg[val_indices]
        eye_val = eye[val_indices]
        pps_val = pps[val_indices]
        arousal_val = arousal[val_indices]
        valence_val = valence[val_indices]

        eeg_test = eeg[test_indices]
        eye_test = eye[test_indices]
        pps_test = pps[test_indices]
        arousal_test = arousal[test_indices]
        valence_test = valence[test_indices]

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
            train_set, val_set, test_set
        )

    def load_data(self):
        # 主加载函数
        raw_data = joblib.load(self.file_path)
        processed_data = self._reshape_eye_pps(raw_data)
        train_set, val_set, test_set = self._get_dataset(processed_data)

        # 创建DataLoader
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    data_loader = CrossSubjectDataLoader(
        file_path=r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl")
    train_loader, val_loader, test_loader = data_loader.load_data()
    # 调试：检查第一个batch的数据形状
    sample_batch = next(iter(train_loader))
    print("Sample batch shapes:")
    for i, data in enumerate(sample_batch):
        print(f"Data {i}: {data.shape if hasattr(data, 'shape') else len(data)}")
