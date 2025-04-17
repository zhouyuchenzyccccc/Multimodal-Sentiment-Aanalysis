import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class Tester:
    def __init__(self, model, test_loader, device='cuda'):
        """
        Args:
            model: 待测试的模型实例
            test_loader: 测试数据加载器
            device: 运行设备
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

        # 测试记录
        self.loss = 0.0
        self.accuracy = 0.0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def load_model(self, model_path):
        """加载预训练模型权重"""
        state_dict = torch.load(model_path, map_location=self.device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")

    def evaluate(self, verbose=True):
        """完整评估流程"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data_dict, labels in tqdm(self.test_loader, desc='Evaluating'):
                # 数据加载
                eeg = data_dict['eeg'].to(self.device).float()
                eye = data_dict['eye'].to(self.device).float()
                pps = data_dict['pps'].to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(eeg, eye, pps)
                probs = torch.softmax(outputs, dim=1)

                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                # 统计预测结果
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # 保存完整结果
                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())

        # 计算总体指标
        self.loss = total_loss / total_samples
        self.accuracy = correct / total_samples

        if verbose:
            self._print_metrics()
            self._plot_confusion_matrix()

        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'predictions': np.array(self.all_preds),
            'labels': np.array(self.all_labels),
            'probabilities': np.array(self.all_probs)
        }

    def _print_metrics(self):
        """打印详细评估指标"""
        print(f"\n{'=' * 40}")
        print(f"Evaluation Results:")
        print(f"- Average Loss: {self.loss:.4f}")
        print(f"- Accuracy: {self.accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(
            self.all_labels,
            self.all_preds,
            target_names=[f'Class {i}' for i in range(len(np.unique(self.all_labels)))]
        ))
        print('=' * 40)

    def _plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(cm.shape[0])],
                    yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def predict_single(self, data_dict):
        """单样本预测"""
        self.model.eval()
        with torch.no_grad():
            eeg = data_dict['eeg'].to(self.device).float().unsqueeze(0)
            eye = data_dict['eye'].to(self.device).float().unsqueeze(0)
            pps = data_dict['pps'].to(self.device).unsqueeze(0)

            outputs = self.model(eeg, eye, pps)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)

        return {
            'prediction': pred.item(),
            'probabilities': probs.squeeze().cpu().numpy()
        }

    def run(self, model_path=None):
        """执行完整测试流程（兼容你的main函数接口）"""
        if model_path is not None:
            self.load_model(model_path)
        return self.evaluate()