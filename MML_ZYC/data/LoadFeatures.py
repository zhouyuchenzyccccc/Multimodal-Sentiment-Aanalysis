# @usage: 读取后的rawData处理成能够训练的形式

import copy
import os
import sys
from pathlib import Path

from tqdm import tqdm


def add_project_root_to_sys_path():
    """动态添加项目根目录到 sys.path"""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


add_project_root_to_sys_path()
import numpy as np
from data.RawData import RawData
from common import utils


class DataFeatures(object):

    def __init__(
            self,
            data_path,
            modalities=['eeg', 'eye', 'au'],
            subject_lists=None,
            Norm=None,
            label_type="",
    ):
        """DataFeatures类用于获取各个模态的初步特征

        DataFeatures加载的简单预处理后的数据，针对不同的模态类型，执行不同的数据预处理策略
        EEG：计算DE特征
        Eye：手动计算眼动特征
        Face：手动计算AU特征

        Args:
            data_path (str): 数据路径，传入pkl所在的路径
            modality (str): 模态名称，需要加载的模态数据，通常包括eeg、eye、au
            subject_lists (list): 受试者列表，shape为(n_trials,)
        Returns:
            data (dict): 数据字典，包含预处理后的数据，label，subject_lists，info等信息
        """
        self.data_path = data_path
        self.subject_lists = subject_lists
        self.ex_nums = None
        rawData = RawData(
            data_path
        )

        self.features = {}
        for modality in modalities:
            # assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
            # self.features[modality] = getattr(self, f"load_{modality}_features")(
            #     rawData.data[modality]
            # )
            # 2025-02-18修改：数据组织形式变化，raw_data存于raw_data字段下，已经手动提取好的特征存于features字段下
            # 注意：Raven数据没有修改数据组织形式，需要用上方注释代码
            if "features" not in rawData.data.keys():
                if "raw_data" in rawData.data.keys():
                    rawData.data.update(rawData.data["raw_data"])
                    del rawData.data["raw_data"]

                assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
                self.features[modality] = getattr(self, f"load_{modality}_features")(
                    rawData.data[modality]
                )
            else:
                assert (
                        modality in rawData.data["features"].keys()
                ), f"数据中不包含{modality}数据"
                feature = copy.deepcopy(
                    rawData.data["features"][modality]
                )
                feature = np.nan_to_num(feature)

                if "au" in modality:
                    feature = AuFeatures._normalize(feature)
                elif "eeg" not in modality:
                    feature = self._normalize(feature)
                self.features[modality] = copy.deepcopy(feature)

                if "eeg" not in modality:
                    self.features[modality] = self.features[modality].reshape(
                        -1, self.features[modality].shape[-1]
                    )

            # if "raw_data" in rawData.data.keys():
            #     rawData.data.update(rawData.data["raw_data"])
            #     del rawData.data["raw_data"]

            # assert modality in rawData.data.keys(), f"数据中不包含{modality}数据"
            # self.features[modality] = getattr(self, f"load_{modality}_features")(
            #     rawData.data[modality]
            # )

            if self.ex_nums is None:
                self.ex_nums = int(
                    self.features[modality].shape[1] // len(self.subject_lists)
                )
                print(self.ex_nums)

            if Norm == "Z_score":
                # 计算均值和标准差
                mean = np.mean(self.features[modality], axis=0)
                std = np.std(self.features[modality], axis=0)
                # 避免除以0，将标准差为0的地方设置为1
                std[std == 0] = 1.0
                # 进行Z-score标准化
                self.features[modality] = (self.features[modality] - mean) / std
            if Norm == "Min_Max":
                self.features[modality] = utils.Min_Max_Normlisze(
                    self.features[modality],
                    sub_nums=len(self.subject_lists),
                    ex_nums=self.ex_nums,
                )
        label_key = "label"
        if label_type != "ruiwen":
            label_key = f"{label_type}_label"
        assert label_key in rawData.data.keys(), f"数据中不包含{label_key}数据"
        if isinstance(rawData.data[label_key], np.ndarray):
            self.label = rawData.data[label_key]
        else:
            self.label = np.concatenate(rawData.data[label_key])

    def _normalize(self, features):
        """
        归一化特征

        Args:
            features (np.ndarray): 特征数组
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 归一化
        features = (features - np.mean(features)) / np.std(features)
        features = (features - features.min()) / (features.max() - features.min())
        return features


class AuFeatures:
    def __init__(self, au_data, subject_lists, data_path):
        """
        FaceFeatures类用于计算和加载面部特征。

        Args:
            au_data (np.ndarray): Face数据，shape为(per_idx, n_trials, n_samples, n_channels)
            subject_lists (list): 受试者列表
            data_path (str): 数据路径
        """
        self.au_data = au_data
        self.subject_lists = subject_lists
        self.data_path = data_path
        self.au_features = None  # 初始化特征缓存

    @staticmethod
    def _normalize(features):
        """
        归一化特征

        Args:
            features (np.ndarray): 特征数组
        Returns:
            np.ndarray: 归一化后的特征
        """
        # 每个AU点有7个特征，共17个AU点
        n_au_points = 17
        features_per_au = 7
        # 对每个AU的7个特征进行独立归一化
        for au_index in range(n_au_points):
            # 获取当前AU的7个特征
            start_idx = au_index * features_per_au
            end_idx = (au_index + 1) * features_per_au
            au_features = features[:, start_idx:end_idx]
            # 归一化
            au_features = (au_features - np.mean(au_features)) / np.std(au_features)
            au_features = (au_features - au_features.min()) / (
                    au_features.max() - au_features.min()
            )
            features[:, start_idx:end_idx] = au_features
        return features

    def compute_au_features(self, feature_dir_name="au_feature"):
        """
        加载或计算面部特征。

        Args:
            feature_dir_name (str): 存储特征的文件夹名称，默认为"au_feature"

        Returns:
            np.ndarray: 处理后的面部特征
        """
        # 获取存储特征的文件夹路径
        au_features_dir = utils.find_nearest_folder(self.data_path)
        au_features_dir = os.path.join(au_features_dir, feature_dir_name)

        # 确保特征目录存在
        if not os.path.exists(au_features_dir):
            raise FileNotFoundError(f"特征目录不存在：{au_features_dir}")

        # 初始化特征列表
        au_track_features = []
        print("开始加载面部特征...")

        for subject in tqdm(self.subject_lists, desc="Processing Subjects"):
            au_feature_path = os.path.join(au_features_dir, f"{subject}.npy")
            if not os.path.exists(au_feature_path):
                raise FileNotFoundError(f"缺少文件：{au_feature_path}")

            # 加载特征文件
            subject_au_features = np.load(au_feature_path)
            au_track_features.append(subject_au_features)

        # 合并所有受试者的特征
        au_track_features = np.concatenate(au_track_features, axis=0)
        au_track_features = np.nan_to_num(au_track_features)  # 替换NaN值
        print("面部特征加载完成。")

        self.au_features = au_track_features
        return au_track_features

    def get_features(self):
        """
        获取计算或加载的面部特征。

        Returns:
            np.ndarray: 面部特征
        """
        if self.au_features is None:
            self.au_features = self.compute_au_features()
        return self.au_features


if __name__ == "__main__":
    # data_path = "/data/Ruiwen/data_with_ICA.pkl"
    # subject_list = [i for i in range(1, 35) if i != 1 and i != 23 and i != 32]
    # print(subject_list)
    # modalities = ["eeg", "eye", "au"]
    # ruiwenData = DataFeatures(
    #     data_path, modalities=modalities, subject_lists=subject_list, Norm="Z_score"
    # )
    # print(ruiwenData.features)
    # for modality in modalities:
    #     print(modality, ruiwenData.features[modality].shape)

    data_path = r"F:\毕业设计\MML_ZYC\HCI_DATA\hci_data.pkl"
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
    print(mahnobData.features)
    for modality in modalities:
        print(modality, mahnobData.features[modality].shape)
    pass
