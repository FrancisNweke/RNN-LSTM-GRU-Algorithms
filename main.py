import torch
import torch.nn as nn
from torchvision import datasets, transforms
from custom_models import RNN, LSTM, GRU
from torch.utils.data import DataLoader
import tqdm
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# region Hyper-parameters
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 3
batch_size = 500
learning_rate = 0.003

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
# endregion

# region MNIST dataset
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())
# endregion

# region Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)
# endregion

# region Initialize model
model = None

print('\nPlease select an algorithm: ')
user_input = input('Press 1 for Recurrent Neural Network (RNN)\nPress 2 for Long Short-Term Memory (LSTM)\nPress 3 '
                   'for Gated Recurrent Unit (GRU) \n---> ')

if user_input == '1':
    print('\nLoading RNN...')
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
elif user_input == '2':
    print('\nLoading LSTM...')
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
elif user_input == '3':
    print('\nLoading GRU...')
    model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
else:
    print(f'You pressed {user_input} which is an invalid input. We will use the default model (RNN) to solve the '
          'problem.')
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# endregion

# region Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# endregion

# region Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# endregion

# Progress bar
print('\nTest in progress...')
for i in tqdm(range(15)):
    time.sleep(1.5)

# region Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
# endregion
