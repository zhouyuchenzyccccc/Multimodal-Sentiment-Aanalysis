import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        # 修改优化器
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=0.0001,  # 减小学习率
                                     weight_decay=0.01)  # 添加L2正则化

        # 定义可学习的对比损失权重
        self.contrastive_weight = nn.Parameter(torch.ones(1, device=device))
        # 将新的权重参数添加到优化器中
        self.optimizer.add_param_group({'params': [self.contrastive_weight]})

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)

        # 训练记录
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        # 早停机制
        self.best_val_loss = float('inf')
        self.patience = 5
        self.counter = 0
        self.early_stop_flag = False

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for data_dict, labels in pbar:
            # 数据预处理
            eeg = data_dict['eeg'].to(self.device).float()
            eye = data_dict['eye'].to(self.device).float()
            pps = data_dict['pps'].to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs, contrastive_loss = self.model(eeg, eye, pps, labels)

            # 添加数值稳定性检查
            if torch.isnan(outputs).any():
                print("Warning: Model output contains NaN!")
                outputs = torch.nan_to_num(outputs)

            # 计算交叉熵损失
            ce_loss = self.criterion(outputs, labels)

            # 修改部分：使用可学习的权重合并损失
            loss = ce_loss + self.contrastive_weight * contrastive_loss

            # 检查损失值
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计信息
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'ce_loss': total_ce_loss / total_samples,
                'contrastive_loss': total_contrastive_loss / total_samples,
                'acc': correct / total_samples
            })

        avg_loss = total_loss / total_samples
        avg_ce_loss = total_ce_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        # 修正准确率计算逻辑
        acc = correct / total_samples if total_samples > 0 else 0.0
        self.train_loss.append(avg_loss)
        self.train_acc.append(acc)
        return avg_loss, avg_ce_loss, avg_contrastive_loss, acc

    def early_stop(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(self.model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {len(self.train_loss)}")
                self.early_stop_flag = True
        return self.early_stop_flag

    def test(self):
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data_dict, labels in self.test_loader:
                eeg = data_dict['eeg'].to(self.device).float()
                eye = data_dict['eye'].to(self.device).float()
                pps = data_dict['pps'].to(self.device)
                labels = labels.to(self.device)

                outputs, contrastive_loss = self.model(eeg, eye, pps, labels)

                # 数值稳定性检查
                if torch.isnan(outputs).any():
                    print("Warning: Test output contains NaN!")
                    outputs = torch.nan_to_num(outputs)

                # 计算交叉熵损失
                ce_loss = self.criterion(outputs, labels)

                # 修改部分：使用可学习的权重合并损失
                loss = ce_loss + self.contrastive_weight * contrastive_loss

                # 跳过无效损失
                if torch.isnan(loss):
                    print("NaN loss in test, skipping batch")
                    continue

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else float('nan')
        avg_contrastive_loss = total_contrastive_loss / total_samples if total_samples > 0 else float('nan')
        # 修正准确率计算逻辑
        acc = correct / total_samples if total_samples > 0 else 0.0
        self.test_loss.append(avg_loss)
        self.test_acc.append(acc)
        return avg_loss, avg_ce_loss, avg_contrastive_loss, acc

    def plot_progress(self):
        plt.figure(figsize=(12, 5))

        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.test_loss, label='Test Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Train Acc')
        plt.plot(self.test_acc, label='Test Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def test_with_loaded_model(self, model_path):
        # 加载模型
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data_dict, labels in self.test_loader:
                eeg = data_dict['eeg'].to(self.device).float()
                eye = data_dict['eye'].to(self.device).float()
                pps = data_dict['pps'].to(self.device)
                labels = labels.to(self.device)

                outputs, contrastive_loss = self.model(eeg, eye, pps, labels)

                # 数值稳定性检查
                if torch.isnan(outputs).any():
                    print("Warning: Test output contains NaN!")
                    outputs = torch.nan_to_num(outputs)

                # 计算交叉熵损失
                ce_loss = self.criterion(outputs, labels)

                # 修改部分：使用可学习的权重合并损失
                loss = ce_loss + self.contrastive_weight * contrastive_loss

                # 跳过无效损失
                if torch.isnan(loss):
                    print("NaN loss in test, skipping batch")
                    continue

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else float('nan')
        avg_ce_loss = total_ce_loss / len(self.test_loader) if len(self.test_loader) > 0 else float('nan')
        avg_contrastive_loss = total_contrastive_loss / len(self.test_loader) if len(self.test_loader) > 0 else float(
            'nan')
        acc = correct / total_samples if total_samples > 0 else 0.0

        print(
            f"Test Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}, Acc: {acc:.4f}")
        return avg_loss, avg_ce_loss, avg_contrastive_loss, acc

    def run(self, epochs, test_person):
        for epoch in range(1, epochs + 1):
            train_loss, train_ce_loss, train_contrastive_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_ce_loss, test_contrastive_loss, test_acc = self.test()

            # 更新学习率
            if not torch.isnan(torch.tensor(test_loss)):
                self.scheduler.step(test_loss)

            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, CE Loss: {train_ce_loss:.4f}, Contrastive Loss: {train_contrastive_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, CE Loss: {test_ce_loss:.4f}, Contrastive Loss: {test_contrastive_loss:.4f}, Acc: {test_acc:.4f}")

            # 早停检查
            if self.early_stop(test_loss):
                # 替换特殊字符以生成合法文件名
                model_name = f"TestPerson{test_person}_epoch{epoch}_TrainLoss{train_loss:.4f}_CELoss{train_ce_loss:.4f}_ContrastiveLoss{train_contrastive_loss:.4f}_Acc{train_acc:.4f}_TestLoss{test_loss:.4f}_CELoss{test_ce_loss:.4f}_ContrastiveLoss{test_contrastive_loss:.4f}_Acc{test_acc:.4f}.pth"
                torch.save(self.model.state_dict(), model_name)
                break

