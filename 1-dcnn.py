import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

heartpy_metrics = pd.read_csv(r'D:\Proga\AML\preprocessed_data\summary.csv')
subject_info = pd.read_csv(r'D:\Proga\AML\panacea\subject-info.csv')

def parse_metrics(metrics_str):
    if 'Could not determine best fit' in metrics_str:
        return None
    
    pattern = r"'(\w+)':\s*(np\.float64\(|)([\d.]+)(\)|)"
    matches = re.findall(pattern, metrics_str)
    
    if not matches:
        return None
    
    metrics_dict = {}
    for key, _, value, _ in matches:
    
        value = float(value)
        metrics_dict[key] = value
    
    return metrics_dict

heartpy_metrics['HeartPy_Metrics'] = heartpy_metrics['HeartPy_Metrics'].apply(parse_metrics)

heartpy_metrics = heartpy_metrics.dropna(subset=['HeartPy_Metrics'])

merged_data = pd.merge(subject_info, heartpy_metrics, on='ID')

unique_age_groups = merged_data['Age_group'].unique()
print(f'Unique Age Groups: {unique_age_groups}')
print(f'Number of Unique Age Groups: {len(unique_age_groups)}')

label_encoder = LabelEncoder()
merged_data['Age_group_encoded'] = label_encoder.fit_transform(merged_data['Age_group'])

encoded_age_groups = merged_data['Age_group_encoded'].unique()
print(f'Encoded Age Groups: {encoded_age_groups}')
print(f'Number of Encoded Age Groups: {len(encoded_age_groups)}')

X = merged_data['HeartPy_Metrics'].apply(lambda x: list(x.values())).values
y = merged_data['Age_group_encoded'].values

X = np.array([np.array(xi) for xi in X])
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class HeartDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HeartDataset(X_train, y_train)
test_dataset = HeartDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (input_length // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_length = X_train.shape[1]
num_classes = len(np.unique(y))

model = CNN1D(input_length, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

example_input = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0)
model.eval()
with torch.no_grad():
    output = model(example_input)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted Age Group: {label_encoder.inverse_transform(predicted.numpy())[0]}, True Age Group: {label_encoder.inverse_transform([y_test[0]])[0]}')