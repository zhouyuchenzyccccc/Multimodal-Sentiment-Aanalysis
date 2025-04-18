import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np


class MultiTaskTrainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # 多任务损失函数
        self.criterion = {
            'arousal': nn.CrossEntropyLoss(),
            'valence': nn.CrossEntropyLoss()
        }

        # 优化器配置
        self.optimizer = optim.AdamW([
            {'params': model.parameters()}  # 自动包含所有模型参数（含contrastive_weight）
        ], lr=1e-4, weight_decay=0.01)

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

        # 训练记录
        self.metrics = {
            'train': {'loss': [], 'a_loss': [], 'v_loss': [], 'c_loss': [], 'a_acc': [], 'v_acc': []},
            'test': {'loss': [], 'a_loss': [], 'v_loss': [], 'c_loss': [], 'a_acc': [], 'v_acc': []}
        }

        # 早停机制
        self.best_val_loss = float('inf')
        self.patience = 5
        self.counter = 0

    def _compute_metrics(self, outputs, labels):
        a_pred = outputs[0].argmax(1)
        v_pred = outputs[1].argmax(1)
        return {
            'a_acc': (a_pred == labels[0]).float().mean().item(),
            'v_acc': (v_pred == labels[1]).float().mean().item()
        }

    def train_epoch(self, epoch):
        self.model.train()
        total = {'loss': 0, 'a_loss': 0, 'v_loss': 0, 'c_loss': 0, 'a_acc': 0, 'v_acc': 0}

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for eeg, eye, pps, arousal, valence in pbar:

            inputs = (
                eeg.to(self.device).float(),
                eye.to(self.device).float(),
                pps.to(self.device).float()
            )
            labels = (
                arousal.to(self.device),
                valence.to(self.device)
            )

            self.optimizer.zero_grad()
            # 前向传播
            a_out, v_out, c_loss = self.model(*inputs, labels=labels)

            # 损失计算
            a_loss = self.criterion['arousal'](a_out, labels[0])
            v_loss = self.criterion['valence'](v_out, labels[1])
            total_loss = a_loss + v_loss + c_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计指标
            metrics = self._compute_metrics((a_out, v_out), labels)
            batch_size = labels[0].size(0)
            total['loss'] += total_loss.item() * batch_size
            total['a_loss'] += a_loss.item() * batch_size
            total['v_loss'] += v_loss.item() * batch_size
            total['c_loss'] += c_loss.item() * batch_size
            total['a_acc'] += metrics['a_acc'] * batch_size
            total['v_acc'] += metrics['v_acc'] * batch_size

            # 进度条更新
            pbar.set_postfix({
                'loss': total['loss'] / ((pbar.n + 1) * batch_size),
                'A/V Acc': f"{metrics['a_acc']:.2%}/{metrics['v_acc']:.2%}"
            })

        # 记录平均指标
        n = len(self.train_loader.dataset)
        self.metrics['train']['loss'].append(total['loss'] / n)
        self.metrics['train']['a_loss'].append(total['a_loss'] / n)
        self.metrics['train']['v_loss'].append(total['v_loss'] / n)
        self.metrics['train']['c_loss'].append(total['c_loss'] / n)
        self.metrics['train']['a_acc'].append(total['a_acc'] / n)
        self.metrics['train']['v_acc'].append(total['v_acc'] / n)

        return {k: v[-1] for k, v in self.metrics['train'].items()}

    def evaluate(self, mode='test'):
        self.model.eval()
        total = {'loss': 0, 'a_loss': 0, 'v_loss': 0, 'c_loss': 0, 'a_acc': 0, 'v_acc': 0}

        loader = self.test_loader if mode == 'test' else self.val_loader
        with torch.no_grad():
            for eeg, eye, pps, arousal, valence in loader:
                inputs = (
                    eeg.to(self.device).float(),
                    eye.to(self.device).float(),
                    pps.to(self.device).float()
                )
                labels = (
                    arousal.to(self.device),
                    valence.to(self.device)
                )

                # 前向传播
                a_out, v_out = self.model(*inputs)
                c_loss = torch.tensor(0).to(self.device)  # 测试时不计算对比损失

                # 损失计算
                a_loss = self.criterion['arousal'](a_out, labels[0])
                v_loss = self.criterion['valence'](v_out, labels[1])
                total_loss = a_loss + v_loss

                # 统计指标
                metrics = self._compute_metrics((a_out, v_out), labels)
                batch_size = labels[0].size(0)
                total['loss'] += total_loss.item() * batch_size
                total['a_loss'] += a_loss.item() * batch_size
                total['v_loss'] += v_loss.item() * batch_size
                total['c_loss'] += c_loss.item() * batch_size
                total['a_acc'] += metrics['a_acc'] * batch_size
                total['v_acc'] += metrics['v_acc'] * batch_size

        # 记录平均指标
        n = len(loader.dataset)
        self.metrics[mode]['loss'].append(total['loss'] / n)
        self.metrics[mode]['a_loss'].append(total['a_loss'] / n)
        self.metrics[mode]['v_loss'].append(total['v_loss'] / n)
        self.metrics[mode]['c_loss'].append(total['c_loss'] / n)
        self.metrics[mode]['a_acc'].append(total['a_acc'] / n)
        self.metrics[mode]['v_acc'].append(total['v_acc'] / n)

        return {k: v[-1] for k, v in self.metrics[mode].items()}

    def early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(self.model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False

    def visualize_progress(self):
        plt.figure(figsize=(15, 6))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train']['loss'], label='Train Loss')
        plt.plot(self.metrics['test']['loss'], label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train']['a_acc'], label='Train Arousal Acc', linestyle='--')
        plt.plot(self.metrics['train']['v_acc'], label='Train Valence Acc', linestyle='--')
        plt.plot(self.metrics['test']['a_acc'], label='Test Arousal Acc')
        plt.plot(self.metrics['test']['v_acc'], label='Test Valence Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run(self, epochs):
        for epoch in range(1, epochs + 1):
            # 训练阶段
            train_metrics = self.train_epoch(epoch)

            # 测试阶段
            test_metrics = self.evaluate()

            # 学习率调整
            self.scheduler.step(test_metrics['loss'])

            # 打印结果
            print(f"\nEpoch {epoch} Results:")
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | A Acc: {train_metrics['a_acc']:.2%} | V Acc: {train_metrics['v_acc']:.2%}")
            print(
                f"Test  Loss: {test_metrics['loss']:.4f} | A Acc: {test_metrics['a_acc']:.2%} | V Acc: {test_metrics['v_acc']:.2%}")

            # # 早停检查
            # if self.early_stopping(test_metrics['loss']):
            #     break

        # 最终可视化
        self.visualize_progress()