import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
%matplotlib inline

# Learning rate
LR = 0.1


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        # FIRST CNN: 1 input image channel, 32 output channels, kernel size 5, activation function ReLu and 2d Max pooling with kernel size 2 and stride 2
        # SECOND CNN: 32 input input channels, 64 output channels, kernel size 5, activation function ReLu and 2d Max pooling with kernel size 2 and stride 2
        self.cnn_model = nn.Sequential(nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2, stride = 2), nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2, stride = 2))
        # Creation of 2 fully connected layers: first layer 1024 input features and 256 output features with activation function Relu, seond layer 256 input features and 10 output
        self.fc_model = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 10), nn.Dropout(0.1))
        
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

# Weights initialization
def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)

# Run on GPU
device = torch.device("cuda:0")
# Create Object
cNet = CNet().to(device)
cNet.apply(weights_init)
# Cross Entropy loss  for classification of multiple classes
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(list(cNet.parameters()), lr = LR)

def evaluation(dataloader):
    total, correct = 0,0
    cNet.eval()
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cNet(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


def main():
    #Train set and test set downloaded
    train_set = datasets.FashionMNIST(root = ".", train = True , download = True , transform = transforms.ToTensor())
    test_set = datasets.FashionMNIST(root = ".", train = False , download = True , transform = transforms.ToTensor())
    # Loading train and test set
    training_loader = torch.utils.data.DataLoader(train_set , batch_size = 32, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_set , batch_size = 32, shuffle = False)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set number of epochs
    nEpochs = 50
    loss_epoch_array = []
    loss_epoch = 0
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(nEpochs):
        loss_epoch = 0
        for i, data in enumerate(training_loader, 0):
            cNet.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = cNet(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            loss_epoch += loss.item()
        loss_epoch_array.append(loss_epoch)
        train_accuracy.append(evaluation(training_loader))
        valid_accuracy.append(evaluation(test_loader))
        print("Epoch {}: loss: {}, train accuracy: {}, valid accuracy:{}".format(epoch + 1, loss_epoch_array[-1], train_accuracy[-1], valid_accuracy[-1]))
    
    # Creates and shows the plot and applies labels
    plt.plot(train_accuracy, label='Training accuracy', color="blue")
    plt.plot(valid_accuracy, label='Validation accuracy', color="pink")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

