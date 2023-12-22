# From here https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Download training data from open datasets.
train_dataset = datasets.CIFAR10(
    train=True,
    root="data",
    download=True,
    transform=ToTensor(),
)

test_dataset = datasets.CIFAR10(
    train=False,
    root="data",
    download=True,
    transform=ToTensor(),
)

class TwoClassDataset(Dataset):
    def __init__(self, original_dataset, class_indices, transform=None):
        self.original_dataset = original_dataset
        self.class_indices = class_indices
        self.filtered_indices = self._filter_indices()

    def _filter_indices(self):
        # Filter indices based on selected classes
        return [idx for idx in range(len(self.original_dataset)) if self.original_dataset[idx][1] in self.class_indices]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Get the item from the original dataset using filtered indices
        original_idx = self.filtered_indices[idx]
        sample, target = self.original_dataset[original_idx]

        # Map the original class indices to new indices
        target = self.class_indices.index(target)
        return sample, target

selected_classes = [1, 3]  

# Create a new dataset with only the selected classes
train_data = TwoClassDataset(train_dataset, selected_classes)
test_data = TwoClassDataset(test_dataset, selected_classes)

x, y = test_data[0][0], test_data[0][1]
print(f"x : {x}, y: {y}")

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
# nn.Module is the Base class for all neural network modules
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# To train the model we need a loss function and an optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Invoke lists for the accuracy and loss values at the end of each epoch
losses, accuracies = [], []
epoch_losses, epoch_accuracies = [], []

# We define training including the predictions and backpropagations

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # why do we need this? 
    model.train() # sets the module in training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Use the loss function to update the weights 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0: # % is remainder 
        # this prints a result every 100 items 
            batch_loss, current = loss.item(), (batch + 1) * len(X)
            print(f"batch number: {batch}, loss: {batch_loss:>7f} [{current:>5d}/{size:>5d}]")
            epoch_losses.append(batch_loss)
    
    epoch_loss = np.array(epoch_losses).mean()

def test(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0 
    with torch.no_grad(): # context manager that disables gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f} \n")


# Conduct training over several epochs
if __name__ == '__main__':
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn, optimizer)
    print("Done!")


    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")



