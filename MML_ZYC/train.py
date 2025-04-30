import torch
import torch.nn.functional as F
from tqdm import tqdm

from MultimodalModel import MultiModalEncoder, ProjectionHead, Classifier

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataLoader.DataLoader import MultimodalDataLoader


def contrastive_loss(z1, z2, labels, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    device = z1.device

    z = torch.cat([z1, z2], dim=0)  # shape: [2B, D]
    sim_matrix = torch.matmul(z, z.T) / temperature  # shape: [2B, 2B]

    # 构造标签和 mask
    labels = labels.view(-1, 1)
    labels = torch.cat([labels, labels], dim=0)  # [2B, 1]

    mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(mask.size(0), dtype=torch.bool).to(device)
    mask = mask.masked_fill(self_mask, 0)  # 排除自身对比

    # softmax 分母
    sim_exp = torch.exp(sim_matrix)
    sim_exp = sim_exp.masked_fill(self_mask, 0)
    sim_sum = sim_exp.sum(dim=1, keepdim=True)

    log_prob = sim_matrix - torch.log(sim_sum + 1e-8)
    loss = - (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return loss.mean()




def contrastive_pretrain_trainer(encoder, projection_head, contrastive_loader, num_epochs=20, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    projection_head.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=lr)

    for epoch in range(num_epochs):
        encoder.train()
        projection_head.train()
        total_loss = 0

        for batch in tqdm(contrastive_loader, desc=f"[Contrastive Epoch {epoch + 1}]"):
            eeg1, eye1, pps1, eeg2, eye2, pps2, labels = [b.to(device) for b in batch]

            # 两个视图分别编码
            e1 = encoder(eeg1, eye1, pps1)
            e2 = encoder(eeg2, eye2, pps2)

            z1 = projection_head(e1)
            z2 = projection_head(e2)

            loss = contrastive_loss(z1, z2,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(contrastive_loader)
        print(f"Epoch [{epoch + 1}] Contrastive Loss: {avg_loss:.4f}")

    return encoder, projection_head


def finetune_trainer(encoder, classifier, train_loader, test_loader, num_epochs=20, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    classifier.to(device)

    # 冻结 encoder（也可以设置成可选参数）
    for param in encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0

        for eeg, eye, pps, arousal, valence in tqdm(train_loader, desc=f"[Finetune Epoch {epoch + 1}]"):
            eeg, eye, pps = eeg.to(device), eye.to(device), pps.to(device)
            arousal, valence = arousal.to(device), valence.to(device)

            features = encoder(eeg, eye, pps).detach()  # 不反向传播
            out_a, out_v = classifier(features)

            loss = criterion(out_a, arousal) + criterion(out_v, valence)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}] Finetune Loss: {total_loss / len(train_loader):.4f}")

        # 测试
        classifier.eval()
        correct_a = correct_v = total = 0
        with torch.no_grad():
            for eeg, eye, pps, arousal, valence in test_loader:
                eeg, eye, pps = eeg.to(device), eye.to(device), pps.to(device)
                arousal, valence = arousal.to(device), valence.to(device)

                features = encoder(eeg, eye, pps)
                out_a, out_v = classifier(features)

                pred_a = out_a.argmax(dim=1)
                pred_v = out_v.argmax(dim=1)

                correct_a += (pred_a == arousal).sum().item()
                correct_v += (pred_v == valence).sum().item()
                total += arousal.size(0)

        print(f"Test Accuracy - Arousal: {correct_a / total:.4f}, Valence: {correct_v / total:.4f}")

    return classifier


if __name__ == "__main__":
    file_path = r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl"
    data_loader = MultimodalDataLoader(
        file_path=r"F:\毕业设计\Multimodal-Sentiment-Aanalysis\MML_ZYC\HCI_DATA\hci_data.pkl")

    subject_list = [1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    all_results = []

    for test_subject_id in subject_list:
        print(f"\n=== Test Subject: {test_subject_id} ===")

        # Step 1: 加载数据
        contrastive_loader, train_loader, test_loader = data_loader.load_data(test_subject_id)

        # Step 2: 初始化模型
        encoder = MultiModalEncoder()
        projection_head = ProjectionHead()
        classifier = Classifier()

        # Step 3: 对比学习预训练
        encoder, projection_head = contrastive_pretrain_trainer(
            encoder, projection_head, contrastive_loader, num_epochs=50,lr=1e-3
        )

        # Step 4: 微调分类器
        classifier = finetune_trainer(
            encoder, classifier, train_loader, test_loader, num_epochs=30, lr=1e-4
        )

        # 评估准确率
        encoder.eval()
        classifier.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        classifier.to(device)

        correct_a = correct_v = total = 0
        with torch.no_grad():
            for eeg, eye, pps, arousal, valence in test_loader:
                eeg, eye, pps = eeg.to(device), eye.to(device), pps.to(device)
                arousal, valence = arousal.to(device), valence.to(device)

                features = encoder(eeg, eye, pps)
                out_a, out_v = classifier(features)

                pred_a = out_a.argmax(dim=1)
                pred_v = out_v.argmax(dim=1)

                correct_a += (pred_a == arousal).sum().item()
                correct_v += (pred_v == valence).sum().item()
                total += arousal.size(0)

        acc_a = correct_a / total
        acc_v = correct_v / total
        all_results.append((test_subject_id, acc_a, acc_v))

        print(f"[Subject {test_subject_id}] Arousal Acc: {acc_a:.4f}, Valence Acc: {acc_v:.4f}")

    print("\n=== Overall Results ===")
    for sid, acc_a, acc_v in all_results:
        print(f"Subject {sid}: Arousal={acc_a:.4f}, Valence={acc_v:.4f}")

    avg_arousal = sum([a for _, a, _ in all_results]) / len(all_results)
    avg_valence = sum([v for _, _, v in all_results]) / len(all_results)
    print(f"\nAverage Arousal Acc: {avg_arousal:.4f}, Average Valence Acc: {avg_valence:.4f}")
