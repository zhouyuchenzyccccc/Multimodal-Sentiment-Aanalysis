import hiddenlayer as hl
from MultimodalModel import MultimodalTransformerModel
import torch
import joblib

from dataLoader.MultimodalDataLoader import MultimodalDataLoader

# # 读取pkl文件
# file_path ="HCI_DATA/hci_data.pkl"
# data = joblib.load(file_path)
#
# # 假设data是字典，且包含'label'键
# if 'label' in data:
#     labels = data['label']
#     # 若labels是列表、数组等可迭代对象，获取唯一的标签类型
#     unique_labels = set(labels)
#     print("data中label的类型有:", unique_labels)
# else:
#     print("data中不包含'label'键。")

loader = MultimodalDataLoader(r"F:\毕业设计\学长代码\Multimodal-sentiment-analysis\MML_ZYC\HCI_DATA\hci_data.pkl")
train, val, test = loader.load_data()

# 检查第一个批次
sample = next(iter(train))
print("\n首个训练批次维度:")
print(f"EEG: {sample[0].shape} (应形如(batch,32,585))")
print(f"Eye: {sample[1].shape} (应形如(batch,38))")
print(f"PPS: {sample[2].shape} (应形如(batch,230))")
print(f"Arousal标签: {sample[3].shape}")
print(f"Valence标签: {sample[4].shape}")

