import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #3 input channels, 16 output, kernel size of 3
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1) #takes the 16 outputs from the first conv as inputs and and outputs 32
        self.fc = nn.Linear(2048, 10) #maps the output to the cifar-10 classes.

    def forward(self,x):
        #push through both conv layers,use the activation function, max pool 
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  
        #flatten to 2048 features
        x = x.view(-1, 2048)
        #pass through fully connected layer
        x = self.fc(x)
        return x

def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 10

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs,labels in trainloader:
            opt.zero_grad()
            outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs,labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")
    print("Training complete.")
    torch.save(model.state_dict(), "model.pth")
    return model
