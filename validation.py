import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('/Users/risapandey/Desktop/SAMSUNG/ser/classified_data.csv')

# Step 2: Preprocess the data
# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Ethnicity', 'Age_Range'], drop_first=True)

# Extract features and labels
feature_columns = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
                   'rolloff', 'zero_crossing_rate'] + [f'mfcc{i}' for i in range(1, 21)]

# Normalize the features
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Prepare the features and labels
X = data[feature_columns].values  # Convert DataFrame to numpy array
y = data['mood'].values  # Labels

# Convert categorical mood labels to numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Transform mood labels to integers

# Reshape X to make it compatible with CNN (e.g., 1D to 2D, [samples, features, 1, 1])
X = X.reshape(X.shape[0], 1, len(feature_columns), 1)  # (N_samples, Channels, Features, Height)

# Convert to PyTorch tensors
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 3: Define the CNN model (Same as before)
class SER_CNN(nn.Module):
    def __init__(self):
        super(SER_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 1), stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1_input_size = self._get_conv_output_size((1, 1, len(feature_columns), 1))  # Dummy input
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, len(label_encoder.classes_))

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            output = self.conv1(torch.ones(shape))
            output = self.pool(output)
            return output.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Initialize the model, criterion, and optimizer
model = SER_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Step 5: Training loop with validation and test accuracy
n_epochs = 30
batch_size = 64

# Lists to store loss and accuracy for each epoch
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_accuracies = []

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training Loop
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(X_train)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation Loop
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            val_inputs = X_val[i:i+batch_size]
            val_labels = y_val[i:i+batch_size]

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_loss = val_running_loss / len(X_val)
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Test accuracy calculation at each epoch
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs, 1)
        test_total = y_test.size(0)
        test_correct = (test_predicted == y_test).sum().item()

    test_accuracy = test_correct / test_total
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch+1}/{n_epochs}, Train Acc: {train_accuracy * 100:.2f}%, '
          f'Val Acc: {val_accuracy * 100:.2f}%, Test Acc: {test_accuracy * 100:.2f}%')

# Step 6: Plotting Accuracy over Epochs
plt.figure(figsize=(12, 5))

# Training, Validation, and Test Accuracy Plot
plt.plot(range(1, n_epochs+1), train_accuracies, label="Training Accuracy", color='green')
plt.plot(range(1, n_epochs+1), val_accuracies, label="Validation Accuracy", color='orange')
plt.plot(range(1, n_epochs+1), test_accuracies, label="Test Accuracy", color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
