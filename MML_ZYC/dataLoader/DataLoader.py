import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from data.LoadFeatures import DataFeatures
import random

class MultimodalDataLoader:
    def __init__(self, file_path, batch_size=64):
        self.file_path = file_path
        self.batch_size = batch_size
        self.subject_lists = [1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]

    def _reshape_eye_pps(self, data):
        data_path = self.file_path
        modalities = ["eeg", "eye", "pps"]
        mahnobData = DataFeatures(
            data_path,
            modalities=modalities,
            subject_lists=self.subject_lists,
            Norm="Z_score",
            label_type="arousal",
        )
        eeg = mahnobData.features["eeg"]
        eye = mahnobData.features["eye"]
        pps = mahnobData.features["pps"]

        return {
            'eeg': eeg,
            'eye': eye,
            'pps': pps,
            'arousal': data['arousal_label'],
            'valence': data['valence_label']
        }

    def _get_dataset(self, data_dict, test_subject_id):
        eeg = data_dict['eeg'].astype(np.float32)
        eye = data_dict['eye'].astype(np.float32)
        pps = data_dict['pps'].astype(np.float32)
        arousal = data_dict['arousal'].astype(np.int64)
        valence = data_dict['valence'].astype(np.int64)

        subject_ids = np.repeat(np.arange(len(self.subject_lists)), 20)

        test_mask = (subject_ids == self.subject_lists.index(test_subject_id))
        train_mask = ~test_mask

        eeg_train, eeg_test = eeg[train_mask], eeg[test_mask]
        eye_train, eye_test = eye[train_mask], eye[test_mask]
        pps_train, pps_test = pps[train_mask], pps[test_mask]
        arousal_train, arousal_test = arousal[train_mask], arousal[test_mask]
        valence_train, valence_test = valence[train_mask], valence[test_mask]
        subject_train = subject_ids[train_mask]

        train_set = TensorDataset(
            torch.from_numpy(eeg_train),
            torch.from_numpy(eye_train),
            torch.from_numpy(pps_train),
            torch.from_numpy(arousal_train),
            torch.from_numpy(valence_train),
            torch.from_numpy(subject_train)
        )

        test_set = TensorDataset(
            torch.from_numpy(eeg_test),
            torch.from_numpy(eye_test),
            torch.from_numpy(pps_test),
            torch.from_numpy(arousal_test),
            torch.from_numpy(valence_test)
        )

        return train_set, test_set

    def _build_contrastive_pairs(self, train_set):
        eeg, eye, pps, arousal, valence, subject = [d.numpy() for d in train_set.tensors]

        eeg1_list, eye1_list, pps1_list = [], [], []
        eeg2_list, eye2_list, pps2_list = [], [], []
        label_list = []

        total_samples = len(subject)
        for subj in np.unique(subject):
            subj_mask = subject == subj
            indices = np.where(subj_mask)[0]

            # 分别收集正/负样本对
            positive_pairs = []
            negative_pairs = []

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    if arousal[idx1] == arousal[idx2] and valence[idx1] == valence[idx2]:
                        positive_pairs.append((idx1, idx2))
                    else:
                        negative_pairs.append((idx1, idx2))

            # 如果某类为空，跳过该被试
            if len(positive_pairs) == 0 or len(negative_pairs) == 0:
                continue

            # 随机采样相同数量的正负样本
            num_keep = min(len(positive_pairs), len(negative_pairs))
            selected_pos = random.sample(positive_pairs, num_keep)
            selected_neg = random.sample(negative_pairs, num_keep)
            selected_pairs = selected_pos + selected_neg
            random.shuffle(selected_pairs)

            for idx1, idx2 in selected_pairs:
                label = 1 if arousal[idx1] == arousal[idx2] and valence[idx1] == valence[idx2] else 0

                eeg1_list.append(eeg[idx1])
                eye1_list.append(eye[idx1])
                pps1_list.append(pps[idx1])

                eeg2_list.append(eeg[idx2])
                eye2_list.append(eye[idx2])
                pps2_list.append(pps[idx2])

                label_list.append(label)

        # 转为 tensor
        eeg1_tensor = torch.tensor(np.stack(eeg1_list), dtype=torch.float32)
        eye1_tensor = torch.tensor(np.stack(eye1_list), dtype=torch.float32)
        pps1_tensor = torch.tensor(np.stack(pps1_list), dtype=torch.float32)
        eeg2_tensor = torch.tensor(np.stack(eeg2_list), dtype=torch.float32)
        eye2_tensor = torch.tensor(np.stack(eye2_list), dtype=torch.float32)
        pps2_tensor = torch.tensor(np.stack(pps2_list), dtype=torch.float32)
        label_tensor = torch.tensor(label_list, dtype=torch.float32)

        contrastive_dataset = TensorDataset(
            eeg1_tensor, eye1_tensor, pps1_tensor,
            eeg2_tensor, eye2_tensor, pps2_tensor,
            label_tensor
        )


        return contrastive_dataset

    def load_data(self, test_subject_id):
        raw_data = joblib.load(self.file_path)
        processed_data = self._reshape_eye_pps(raw_data)
        train_set, test_set = self._get_dataset(processed_data, test_subject_id)

        # 构建对比学习数据集
        contrastive_set = self._build_contrastive_pairs(train_set)
        contrastive_loader = DataLoader(contrastive_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # 原始训练/测试集移除 subject_id
        cleaned_train_set = TensorDataset(*train_set.tensors[:5])
        train_loader = DataLoader(cleaned_train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        return contrastive_loader, train_loader, test_loader



# 使用示例
if __name__ == "__main__":
    data_loader = MultimodalDataLoader(
        file_path=r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl")

    subject_lists = [1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]

    for test_subject_id in subject_lists:
        print(f"Training with subject {test_subject_id} as test set")
        contrastive_loader,train_loader, test_loader = data_loader.load_data(test_subject_id)

        # 调试：检查第一个batch的数据形状
        sample_batch = next(iter(train_loader))
        print("Train Sample batch shapes:")
        for i, data in enumerate(sample_batch):
            print(f"Data {i}: {data.shape if hasattr(data, 'shape') else len(data)}")

        sample_batch = next(iter(test_loader))
        print("Test Sample batch shapes:")
        for i, data in enumerate(sample_batch):
            print(f"Data {i}: {data.shape if hasattr(data, 'shape') else len(data)}")
