import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = r"D:\NTU CE\1132_Deep Learning\HW4\damage_classification_forTrain\beam_damage"
batch_size, epochs, lr = 32, 500, 1e-4
patience = 10
stopping_counter = 0
best_acc = 0

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(train_dir, transform=transform)
class_names = dataset.classes  # ['Class A', 'Class B', 'Class C']
num_classes = len(class_names)
train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
in_feats = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_feats, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total, correct = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"[Damage] Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    acc = correct / total * 100

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = out.argmax(1)
            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total * 100

    print(f"Epoch {epoch+1}: Train Acc={acc:.2f}%, Val Acc={val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_damage_model.pth")
        stopping_counter = 0
    else:
        stopping_counter += 1
        print(f"  No improvement for {stopping_counter} epoch(s).")
        if stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (no improve for {patience} epochs).")
            break

print("Best val acc (damage):", best_acc)