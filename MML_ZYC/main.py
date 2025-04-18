import yaml
from MultimodalModel import MultimodalTransformerModel
from Tester import Tester
import torch
import joblib
from data.Dataset import load_data, FeatureDataset
from dataLoader.MultiTaskTrainer import MultiTaskTrainer
from dataLoader.MultimodalDataLoader import MultimodalDataLoader


def load_config(config_path="config/config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def run(config, model):
    # 初始化数据加载器
    data_loader = MultimodalDataLoader(file_path="HCI_DATA/hci_data.pkl")
    train_loader, val_loader, test_loader = data_loader.load_data()

    # 创建训练器
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 开始训练
    trainer.run(
        epochs=config["training"]["epochs"],
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

    # 初始化模型
    model = MultimodalTransformerModel()

    # train_loader, test_loader = load_data(config, test_person=1)
    # print(train_loader.__len__())
    # print(test_loader.__len__())


    # 2. 训练模型
    run(config, model)

    # for test_person in [17, 18, 19, 20]:
    #     test(config, test_person,
    #          model_path="models_2/TestPerson13_epoch12_TrainLoss0.0237_CELoss0.0018_ContrastiveLoss0.0221_Acc0.9775_TestLoss0.1573_CELoss0.0026_ContrastiveLoss0.1558_Acc1.0000.pth")
