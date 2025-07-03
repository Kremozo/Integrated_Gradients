from SimpleCNN import train_model
from SimpleCNN import SimpleCNN
from IntegratedGradients import integrated_gradients
from Visualizer import visualize_attributions

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random

def main():
    model = SimpleCNN()
    try: 
        model.load_state_dict(torch.load('model.pth'))
        print("Loaded trained state")
    except FileNotFoundError:
        print("no model state found, training")
        model = train_model()
    
    model.eval

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    random_idx = random.randint(0, len(testset)-1)

    img_tensor, label = testset[1]
    img_tensor = img_tensor.unsqueeze(0)

    original_img = testset.data[1]
    original_img = Image.fromarray(original_img)

    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        print(f"Predicted class: {pred_class}, True class: {label}")

    attributions = integrated_gradients(img_tensor, model, pred_class, baseline=torch.zeros_like(img_tensor), steps=50)

    visualize_attributions(original_img, attributions)

if __name__ == "__main__":
    main()
    