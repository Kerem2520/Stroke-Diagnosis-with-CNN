import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

veri_yolu = "C:/Users/musta/Desktop/YapayZekaDersi/Egitim_Test"
siniflar = ["inme_yok", "iskemik", "kanamali"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


BATCH_SIZE = 16

class MRDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for i, sinif in enumerate(classes):
            sinif_dizin = os.path.join(root_dir, sinif)
            for dosya in os.listdir(sinif_dizin):
                self.image_paths.append(os.path.join(sinif_dizin, dosya))
                self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = MRDataset(veri_yolu, siniflar, transform)
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.25, random_state=42, stratify=dataset.labels)
train_set = torch.utils.data.Subset(dataset, train_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model_resnet18 = models.resnet18(weights="IMAGENET1K_V1")
for param in model_resnet18.parameters():
    param.requires_grad = True

num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, len(siniflar))

for param in model_resnet18.fc.parameters():
    param.requires_grad = True

model_resnet18 = model_resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_resnet18.parameters()), lr=0.001, momentum=0.9)

num_epochs = 10

epoch_list = []
train_acc_list = []
train_f1_list = []
train_loss_list = []
test_acc_list = []
test_f1_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model_resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    train_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    model_resnet18.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_resnet18(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_all_labels.extend(labels.cpu().numpy())
            test_all_predictions.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    test_acc = 100 * test_correct / test_total
    test_loss_avg = test_loss / len(test_loader)
    test_f1 = f1_score(test_all_labels, test_all_predictions, average='weighted')

    epoch_list.append(epoch + 1)
    train_acc_list.append(train_acc)
    train_f1_list.append(f1 * 100)
    train_loss_list.append(train_loss * 100)
    test_acc_list.append(test_acc)
    test_f1_list.append(test_f1 * 100)
    test_loss_list.append(test_loss_avg * 100)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, F1 Score: {f1:.4f}, "
          f"Test Loss: {test_loss_avg:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}")


torch.save(model_resnet18.state_dict(), 'YapayZekaModel4.pth')

plt.figure(figsize=(12, 8))
plt.plot(epoch_list, train_acc_list, label='Train Accuracy (%)')
plt.plot(epoch_list, test_acc_list, label='Test Accuracy (%)')
#plt.plot(epoch_list, train_f1_list, label='Train F1 Score (%)')
#plt.plot(epoch_list, test_f1_list, label='Test F1 Score (%)')
plt.plot(epoch_list, train_loss_list, label='Train Loss (%)')
plt.plot(epoch_list, test_loss_list, label='Test Loss (%)')
plt.xlabel('Epoch')
plt.ylabel('Yüzde (%)')
plt.title('Model Performansı')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


cm = confusion_matrix(test_all_labels, test_all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=siniflar, yticklabels=siniflar, cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()
