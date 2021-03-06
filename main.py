from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.functional import relu as Relu
from tqdm import tqdm
# train settings
batch_size = 64

# MNIST Datasets
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# Use PyTorch Dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(5, 20, kernel_size=5)
        self.l1 = nn.Linear(320, 180)
        self.l2 = nn.Linear(180, 60)
        self.dropout = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(60, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = Relu(self.mp(self.conv1(x)))
        x = Relu(self.mp(self.conv2(x)))
        x = x.view(input_size, -1)  # Flatten the tensor
        x = Relu(self.l1(x))
        x = Relu(self.l2(x))
        # no need for activation function because using CrossEntropyLoss
        return self.dropout(self.l3(x))


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.5)


def train(epoch):
    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        result = model(data)
        loss = criterion(result, target)
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        result = model(data)
        test_loss += criterion(result, target).item()
        prediction = result.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

    correct = 100.0 * correct / len(test_loader.dataset)
    print(f'Correct Rate: {correct:.4f}%')


if __name__ == "__main__":
    for epoch in range(1, 21):
        train(epoch)

test()
