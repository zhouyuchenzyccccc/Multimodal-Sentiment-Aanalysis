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
        # 初始化各阶段组件
        self.phase1_optimizer = None
        self.phase1_scheduler = None
        self.phase2_optimizer = None
        self.phase2_scheduler = None
        self.phase3_optimizer = None
        self.phase3_scheduler = None
        # 多任务损失函数
        self.criterion = {
            'arousal': nn.CrossEntropyLoss(),
            'valence': nn.CrossEntropyLoss()
        }

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

    def _freeze_all(self):
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False

    def _setup_phase1(self):
        """阶段1：特征提取参数优化"""
        self._freeze_all()
        # 解冻特征提取网络
        for module in [self.model.eeg_net, self.model.eye_net, self.model.pps_net]:
            for param in module.parameters():
                param.requires_grad = True
        # 创建阶段专属优化器和调度器
        self.phase1_optimizer = optim.AdamW(
            list(self.model.eeg_net.parameters()) +
            list(self.model.eye_net.parameters()) +
            list(self.model.pps_net.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        self.phase1_scheduler = ReduceLROnPlateau(
            self.phase1_optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )
        return self.phase1_optimizer

    def _setup_phase2(self):
        """阶段2：融合模块参数优化"""
        self._freeze_all()
        # 解冻融合相关模块
        modules_to_unfreeze = [
            self.model.cross_attn_e2p,
            self.model.cross_attn_p2e,
            self.model.attention_weights,
            self.model.fusion,
            self.model.arousal_head
        ]
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
        # 创建阶段专属优化器和调度器
        self.phase2_optimizer = optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=1e-4,
            weight_decay=0.01
        )
        self.phase2_scheduler = ReduceLROnPlateau(
            self.phase2_optimizer,
            mode='min',
            patience=2,
            factor=0.2
        )
        return self.phase2_optimizer

    def _setup_phase3(self):
        """阶段3：Valence头参数优化"""
        self._freeze_all()
        for param in self.model.valence_head.parameters():
            param.requires_grad = True
        # 创建阶段专属优化器和调度器
        self.phase3_optimizer = optim.AdamW(
            self.model.valence_head.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.phase3_scheduler = ReduceLROnPlateau(
            self.phase3_optimizer,
            mode='min',
            patience=2,
            factor=0.1
        )
        return self.phase3_optimizer

    def train_epoch_phase1(self, epoch):  # 训练特征提取模块和对比损失
        self.model.train()
        optimizer = self._setup_phase1()

        total = {'loss': 0, 'a_loss': 0, 'v_loss': 0, 'c_loss': 0, 'a_acc': 0, 'v_acc': 0}

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch} (Phase 1)')
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

            optimizer.zero_grad()
            # 前向传播
            a_out, v_out, c_loss = self.model(*inputs, labels=labels)

            # 损失计算
            total_loss = 0.8 * c_loss + 0.1 * self.criterion['arousal'](a_out, labels[0]) + 0.1 * self.criterion[
                'valence'](v_out, labels[1])
            # ---------------------------------------------------------------------
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # 统计指标
            metrics = self._compute_metrics((a_out, v_out), labels)
            batch_size = labels[0].size(0)
            total['loss'] += total_loss.item() * batch_size
            total['a_loss'] += 0  # 不计算arousal loss
            total['v_loss'] += 0  # 不计算valence loss
            total['c_loss'] += c_loss.item() * batch_size
            total['a_acc'] += metrics['a_acc'] * batch_size
            total['v_acc'] += metrics['v_acc'] * batch_size

            # 进度条更新
            pbar.set_postfix({
                'loss': total['loss'] / ((pbar.n + 1) * batch_size),
                'C Loss': c_loss.item()
            })

        # 记录平均指标
        n = len(self.train_loader.dataset)
        self.metrics['train']['loss'].append(total['loss'] / n)
        self.metrics['train']['a_loss'].append(0)  # 不计算arousal loss
        self.metrics['train']['v_loss'].append(0)  # 不计算valence loss
        self.metrics['train']['c_loss'].append(total['c_loss'] / n)
        self.metrics['train']['a_acc'].append(total['a_acc'] / n)
        self.metrics['train']['v_acc'].append(total['v_acc'] / n)

        return {k: v[-1] for k, v in self.metrics['train'].items()}

    def train_epoch_phase2(self, epoch):  # 训练融合模块和arousal_head
        self.model.train()

        # 创建新的优化器，只包含融合模块和arousal_head的参数
        fusion_optimizer = self._setup_phase2()

        total = {'loss': 0, 'a_loss': 0, 'v_loss': 0, 'c_loss': 0, 'a_acc': 0, 'v_acc': 0}

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch} (Phase 2)')
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

            fusion_optimizer.zero_grad()
            # 前向传播
            a_out, v_out, c_loss = self.model(*inputs, labels=labels)

            # 损失计算
            a_loss = self.criterion['arousal'](a_out, labels[0])

            total_loss = a_loss+c_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            fusion_optimizer.step()

            # 统计指标
            metrics = self._compute_metrics((a_out, v_out), labels)
            batch_size = labels[0].size(0)
            total['loss'] += total_loss.item() * batch_size
            total['a_loss'] += a_loss.item() * batch_size
            total['v_loss'] += 0  # 不计算valence loss
            total['c_loss'] += 0  # 不计算对比损失
            total['a_acc'] += metrics['a_acc'] * batch_size
            total['v_acc'] += metrics['v_acc'] * batch_size

            # 进度条更新
            pbar.set_postfix({
                'loss': total['loss'] / ((pbar.n + 1) * batch_size),
                'A Acc': f"{metrics['a_acc']:.2%}"
            })

        # 记录平均指标
        n = len(self.train_loader.dataset)
        self.metrics['train']['loss'].append(total['loss'] / n)
        self.metrics['train']['a_loss'].append(total['a_loss'] / n)
        self.metrics['train']['v_loss'].append(0)  # 不计算valence loss
        self.metrics['train']['c_loss'].append(0)  # 不计算对比损失
        self.metrics['train']['a_acc'].append(total['a_acc'] / n)
        self.metrics['train']['v_acc'].append(total['v_acc'] / n)

        return {k: v[-1] for k, v in self.metrics['train'].items()}

    def train_epoch_phase3(self, epoch):  # 训练Valence分类头
        self.model.train()

        # 创建新的优化器，只包含valence_head的参数
        valence_optimizer = self._setup_phase3()

        total = {'loss': 0, 'a_loss': 0, 'v_loss': 0, 'c_loss': 0, 'a_acc': 0, 'v_acc': 0}

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch} (Phase 3)')
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

            valence_optimizer.zero_grad()
            # 前向传播
            a_out, v_out, c_loss = self.model(*inputs, labels=labels)

            # 损失计算
            v_loss = self.criterion['valence'](v_out, labels[1])
            total_loss = v_loss

            v_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            valence_optimizer.step()

            # 统计指标
            metrics = self._compute_metrics((a_out, v_out), labels)
            batch_size = labels[0].size(0)
            total['loss'] += total_loss.item() * batch_size
            total['a_loss'] += 0  # 不计算arousal loss
            total['v_loss'] += v_loss.item() * batch_size
            total['c_loss'] += 0  # 不计算对比损失
            total['a_acc'] += metrics['a_acc'] * batch_size
            total['v_acc'] += metrics['v_acc'] * batch_size

            # 进度条更新
            pbar.set_postfix({
                'loss': total['loss'] / ((pbar.n + 1) * batch_size),
                'V Acc': f"{metrics['v_acc']:.2%}"
            })

        # 记录平均指标
        n = len(self.train_loader.dataset)
        self.metrics['train']['loss'].append(total['loss'] / n)
        self.metrics['train']['a_loss'].append(0)  # 不计算arousal loss
        self.metrics['train']['v_loss'].append(total['v_loss'] / n)
        self.metrics['train']['c_loss'].append(0)  # 不计算对比损失
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
                a_out, v_out, c_loss = self.model(*inputs, labels=labels)
                # c_loss = torch.tensor(0).to(self.device)  # 测试时不计算对比损失

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

    def run(self, epochs_phase1, epochs_phase2, epochs_phase3):
        print("Phase 1: Training Feature Extractors with Contrastive Loss")
        for epoch in range(1, epochs_phase1 + 1):
            # 训练阶段1
            train_metrics = self.train_epoch_phase1(epoch)

            # 测试阶段
            test_metrics = self.evaluate()

            # 学习率调整
            self.phase1_scheduler.step(test_metrics['loss'])  # 使用对应阶段的损失

            # 打印结果
            print(f"\nEpoch {epoch} Results:")
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | A Acc: {train_metrics['a_acc']:.2%} | V Acc: {train_metrics['v_acc']:.2%} | C Loss: {train_metrics['c_loss']:.4f}")
            print(
                f"Test  Loss: {test_metrics['loss']:.4f} | A Acc: {test_metrics['a_acc']:.2%} | V Acc: {test_metrics['v_acc']:.2%}")

            # # 早停检查
            # if self.early_stopping(test_metrics['loss']):
            #     break

        print("\nPhase 2: Training Fusion Module and Arousal Head")
        for epoch in range(1, epochs_phase2 + 1):
            # 训练阶段2
            train_metrics = self.train_epoch_phase2(epoch)

            # 测试阶段
            test_metrics = self.evaluate()

            # 学习率调整
            self.phase2_scheduler.step(test_metrics['loss'])  # 使用对应阶段的损失

            # 打印结果
            print(f"\nEpoch {epoch} Results:")
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | A Acc: {train_metrics['a_acc']:.2%} | V Acc: {train_metrics['v_acc']:.2%}")
            print(
                f"Test  Loss: {test_metrics['loss']:.4f} | A Acc: {test_metrics['a_acc']:.2%} | V Acc: {test_metrics['v_acc']:.2%}")

            # # 早停检查
            # if self.early_stopping(test_metrics['loss']):
            #     break

        print("\nPhase 3: Training Valence Head Only")
        for epoch in range(1, epochs_phase3 + 1):
            # 训练阶段3
            train_metrics = self.train_epoch_phase3(epoch)

            # 测试阶段
            test_metrics = self.evaluate()

            # 学习率调整
            self.phase3_scheduler.step(test_metrics['loss'])  # 使用对应阶段的损失

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
