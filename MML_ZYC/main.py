import yaml
from MultimodalModel import MultimodalTransformerModel
from Tester import Tester
import torch
import joblib
from data.Dataset import load_data, FeatureDataset
from dataLoader.CrossSubjectDataLoader import CrossSubjectDataLoader
from dataLoader.DataLoader import MultimodalDataLoader
from dataLoader.MultiTaskTrainer import MultiTaskTrainer


def load_config(config_path="config/config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def run(config, model, train_loader, test_loader, test_person):
    # 创建训练器
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        test_person=test_person
    )
    # 开始训练
    trainer.run(
        50, 70, 50, 10, 100
    )


def test(config, test_person, model_path):
    """测试主函数"""
    # 1. 加载数据
    train_loader, test_loader = load_data(config, test_person=test_person)

    # 2. 初始化模型
    model = MultimodalTransformerModel()

    # 3. 初始化测试器
    tester = Tester(model, train_loader, device="cuda")

    # 4. 执行评估
    results = tester.run(model_path)

    # 5. 返回结果（可根据需要保存）
    return results


if __name__ == "__main__":
    # 读取pkl文件
    file_path = 'HCI_DATA\hci_data.pkl'
    data = joblib.load(file_path)

    # 1. 加载数据
    config = load_config()
    subject_lists = [1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    data_loader = MultimodalDataLoader(
        file_path=r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl")
    for test_subject_id in subject_lists:
        print(f"Training with subject {test_subject_id} as test set")
        train_loader, test_loader = data_loader.load_data(test_subject_id)
        # 初始化模型
        model = MultimodalTransformerModel()
        # 2. 训练模型
        run(config, model, train_loader, test_loader, test_subject_id)
