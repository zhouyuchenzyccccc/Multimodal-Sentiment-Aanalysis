# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import MultiModalEncoder, ProjectionHead, Classifier
from data_loader import EmotionDataset, ContrastiveDataset, default_augment

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
pretrain_epochs = 50
finetune_epochs = 30
lr_pretrain = 1e-3
lr_finetune = 1e-4
temperature = 0.5

# Load datasets (paths are placeholders; replace with actual file paths)
eeg_path    = 'eeg_data.npy'
eye_path    = 'eye_data.npy'
phy_path    = 'physio_data.npy'
label_path  = 'labels.npy'   # labels shape (N,2), each entry like [arousal_label, valence_label]

dataset = EmotionDataset(eeg_path, eye_path, phy_path, label_path)
# Split for fine-tuning (e.g., 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_base, val_dataset = random_split(dataset, [train_size, val_size])

# Contrastive pre-training uses the entire unlabeled set (we ignore true labels)
contrastive_dataset = ContrastiveDataset(dataset, augment=default_augment)
contrastive_loader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
encoder = MultiModalEncoder().to(device)
projector = ProjectionHead().to(device)
classifier = Classifier().to(device)

# Optimizers
optimizer_pre = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr_pretrain)
optimizer_fine = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr_finetune)

# NT-Xent Contrastive Loss (SimCLR-style)
def contrastive_loss(z1, z2, temperature):
    """
    Compute NT-Xent loss between two batches of representations z1 and z2.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # (2N, dim)
    z = F.normalize(z, dim=1)
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # (2N, 2N)
    # Mask self-similarities
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, -9e15)
    # Similarity divided by temperature
    sim_matrix = sim_matrix / temperature
    # For each i in [0,2N), the positive example index is i+batch if i<batch, else i-batch
    targets = torch.cat([torch.arange(batch_size, 2*batch_size, device=z.device),
                         torch.arange(0, batch_size, device=z.device)], dim=0)
    # Compute cross-entropy loss
    loss = F.cross_entropy(sim_matrix, targets)
    return loss

# 1) Pre-training loop (contrastive learning)
print("Starting contrastive pre-training...")
encoder.train(); projector.train()
for epoch in range(pretrain_epochs):
    total_loss = 0.0
    for (eeg1, eye1, phy1, eeg2, eye2, phy2) in contrastive_loader:
        eeg1 = eeg1.to(device); eye1 = eye1.to(device); phy1 = phy1.to(device)
        eeg2 = eeg2.to(device); eye2 = eye2.to(device); phy2 = phy2.to(device)
        # Forward passes for both views
        h1 = encoder(eeg1, eye1, phy1)   # fused features (batch, feat_dim)
        h2 = encoder(eeg2, eye2, phy2)
        z1 = projector(h1)  # projection head
        z2 = projector(h2)
        # Compute loss
        loss = contrastive_loss(z1, z2, temperature)
        # Backward and optimize
        optimizer_pre.zero_grad()
        loss.backward()
        optimizer_pre.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(contrastive_loader)
    print(f"Epoch [{epoch+1}/{pretrain_epochs}], Contrastive Loss: {avg_loss:.4f}")

# 2) Fine-tuning loop (supervised classification)
print("\nStarting fine-tuning with classifier...")
# Prepare DataLoader for fine-tuning
train_loader = DataLoader(train_base, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(finetune_epochs):
    encoder.train(); classifier.train()
    train_loss = 0.0
    for (eeg, eye, phy, label) in train_loader:
        eeg = eeg.to(device); eye = eye.to(device); phy = phy.to(device)
        # Split labels into arousal and valence
        label = label.to(device)
        label_a = label[:, 0]  # arousal
        label_v = label[:, 1]  # valence
        # Forward
        features = encoder(eeg, eye, phy)
        out_a, out_v = classifier(features)
        # Compute cross-entropy losses for each
        loss_a = F.cross_entropy(out_a, label_a)
        loss_v = F.cross_entropy(out_v, label_v)
        loss = loss_a + loss_v  # combined loss
        # Backward and optimize
        optimizer_fine.zero_grad()
        loss.backward()
        optimizer_fine.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    # Validation (optional)
    encoder.eval(); classifier.eval()
    correct_a = 0; correct_v = 0; total = 0
    with torch.no_grad():
        for (eeg, eye, phy, label) in val_loader:
            eeg = eeg.to(device); eye = eye.to(device); phy = phy.to(device)
            label = label.to(device)
            out_a, out_v = classifier(encoder(eeg, eye, phy))
            _, pred_a = torch.max(out_a, 1)
            _, pred_v = torch.max(out_v, 1)
            correct_a += (pred_a == label[:,0]).sum().item()
            correct_v += (pred_v == label[:,1]).sum().item()
            total += eeg.size(0)
    acc_a = correct_a / total * 100
    acc_v = correct_v / total * 100
    print(f"Epoch [{epoch+1}/{finetune_epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc Arousal: {acc_a:.1f}%, Val Acc Valence: {acc_v:.1f}%")
