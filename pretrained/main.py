import torch
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

#load image
image_path = "cat.jpg"

# imagenet normalize
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#process image
original_img = Image.open(image_path).convert("RGB")
input_tensor = transform(original_img).unsqueeze(0)

#get model's prediction
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class index: {pred_class}")

#calculate Integrated Gradients
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input_tensor,
    baselines=torch.zeros_like(input_tensor),
    target=pred_class,
    return_convergence_delta=True,
    n_steps=50
)

def visualize_attributions(original_img, attributions):
    # Convert attribution to heatmap
    attr = attributions.squeeze().detach().numpy()
    attr = np.sum(np.abs(attr), axis=0)  # combine channels
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-10)  # normalize

    # Resize original image
    original_np = np.asarray(original_img.resize((224, 224))) / 255.0

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    #plt.imshow(original_np)
    plt.imshow(attr, cmap='hot', alpha=0.5)
    plt.title("Integrated Gradients Attribution")
    plt.tight_layout()
    plt.show()

visualize_attributions(original_img, attributions)