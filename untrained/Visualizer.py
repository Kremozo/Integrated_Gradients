from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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
    plt.imshow(attr, cmap='hot', alpha=0.5)
    plt.title("Integrated Gradients Attribution")
    plt.tight_layout()
    plt.show()