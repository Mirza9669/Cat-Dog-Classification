import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load image function
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

# Data paths and ratios
csv_file = r"H:\Internship\Cat & Dog Classification\data.csv"
img_folder = r"H:\Internship\Cat & Dog Classification\PetImages"
train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

# Load and preprocess data
df = pd.read_csv(csv_file)
df = df.sample(frac=1, random_state=1)

train_df, test_df = train_test_split(df, test_size=test_ratio)
train_df, val_df = train_test_split(df, test_size=val_ratio / (train_ratio + val_ratio))

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}

train_images = [load_image(os.path.join(img_folder, 'Dog', f"{img}.jpg"), data_transforms['train']) if img.startswith('d')
                else load_image(os.path.join(img_folder, 'Cat', f"{img}.jpg"), data_transforms['train'])
                for img in train_df.iloc[:, 0]]

val_images = [load_image(os.path.join(img_folder, 'dog', f"{img}.jpg"), data_transforms['val']) if img.startswith('d')
              else load_image(os.path.join(img_folder, 'cat', f"{img}.jpg"), data_transforms['val'])
              for img in val_df.iloc[:, 0]]

test_images = [load_image(os.path.join(img_folder, 'dog', f"{img}.jpg"), data_transforms['test']) if img.startswith('d')
                else load_image(os.path.join(img_folder, 'cat', f"{img}.jpg"), data_transforms['test'])
                for img in test_df.iloc[:, 0]]

label_mapping = {"Cat": 0, "Dog": 1}
print(label_mapping)

train_label = [label_mapping[label] for label in train_df.iloc[:, 1].tolist()]
val_label = [label_mapping[label] for label in val_df.iloc[:, 1].tolist()]
test_label = [label_mapping[label] for label in test_df.iloc[:, 1].tolist()]

# Define datasets and dataloaders
train_dataset = CustomDataset(train_images, train_label, transform=None)  # Transforms already applied
val_dataset = CustomDataset(val_images, val_label, transform=None)
test_dataset = CustomDataset(test_images, test_label, transform=None)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Define Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(3, out_channels= 32, kernel_size=3),
            # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
            # output = ((224 - 3 + 2 * 0) / 1) + 1 = 222
            # shape = 32x222x222
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # output = 222 / 2 = 111
            # shape = 32x111x111
            
            nn.Conv2d(32, out_channels= 64, kernel_size=3),
            # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
            # output = ((111 - 3 + 2 * 0) / 1) + 1 = 109
            # shape = 64x109x109
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output = 109 / 2 = 54
            # shape = 64x54x54
            
# =============================================================================
#             nn.Conv2d(64, out_channels= 128, kernel_size= 3),
#             # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
#             # output = ((54 - 3 + 2 * 0) / 1) + 1 = 52
#             # shape = 128x52x52
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size= 2, stride= 2),
#             # output = 52 / 2 = 26
#             # shape = 128x26x26
# =============================================================================
            
# =============================================================================
#             nn.Conv2d(128, out_channels= 256, kernel_size=3, stride=1, padding="same"),
#             # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
#             # output = ((26 - 3 + 2 * 0) / 1) + 1 = 24
#             # shape = 256x24x24
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # output = 24 / 2 = 12
#             # shape = 256x12x12
# =============================================================================
            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 54 * 54, 224),  
            nn.ReLU(),
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Instantiate the model
num_classes = len(label_mapping)
model = SimpleCNN(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
count = 0

# Training on training set
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        print(f"{count} Cycle Completed")
        count+=1
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
    count=0
    
# Evaluation on validation set
model.eval()
correct = 0
total = 0
count = 0

with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"{count} Cycle Completed")
        count+=1

val_accuracy = correct / total
print(f"Validation Accuracy: {val_accuracy:.2%}")

# Evaluation on test set
model.eval()
all_predictions = []
all_labels = []
count = 0 

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"{count} Cycle Completed")
        count+=1
        
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.2%}")

# Calculate metrics for each class
class_names = list(label_mapping.keys())
test_f1_scores = f1_score(all_labels, all_predictions, average=None, labels=[0, 1])
test_precision_scores = precision_score(all_labels, all_predictions, average=None, labels=[0, 1])
test_recall_scores = recall_score(all_labels, all_predictions, average=None, labels=[0, 1])

# Calculate metrics
test_f1_score = f1_score(all_labels, all_predictions, average='weighted')
test_precision = precision_score(all_labels, all_predictions, average='weighted')
test_recall = recall_score(all_labels, all_predictions, average='weighted')

print(f"Test F1 Score: {test_f1_score:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"  F1 Score: {test_f1_scores[i]:.4f}")
    print(f"  Precision: {test_precision_scores[i]:.4f}")
    print(f"  Recall: {test_recall_scores[i]:.4f}")

conf_matrix = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(conf_matrix)

torch.save(model.state_dict(), 'cat-dog-classification.h5')