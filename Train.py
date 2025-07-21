import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from Class_dataset import FaceDataset
from Model_Resnet import FaceNetEmbedding
from Loss_Function import TripletLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform với Augmentation ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# === Load dữ liệu ===
train_dataset = FaceDataset(root_dir='Dataset_1/train', transform=transform)
val_dataset = FaceDataset(root_dir='Dataset_1/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# === Model, Loss, Optimizer ===
model = FaceNetEmbedding(embedding_size=128).to(device)
criterion = TripletLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        anchor_emb = model(anchor)
        pos_emb = model(positive)
        neg_emb = model(negative)

        loss = criterion(anchor_emb, pos_emb, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print("✅ Saved best model.")

print("✅ Training completed.")
