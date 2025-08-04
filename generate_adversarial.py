import torch
import numpy as np
from feature_extraction import *
from torchvision import models, transforms

"""
This function adds the adversarial noise to the input image
Then tries to optimise the features vector so that it is close to original input image

"""
def generate_adversarial(input_image, desired_label, epsilon=0.03, alpha=0.05, steps=100, lambda_perceptual=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.vgg16(pretrained=True).to(device).eval()
    # encoder = model.features[:23].eval()
    # encoder.to(device)
    # Build encoder: conv layers up to last pooling, then avg pool
    encoder = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(start_dim=1, end_dim=2)   # yields [1,512]
    ).to(device).eval()
   
    input_image = load_image(input_image, device)
    input_image = input_image.requires_grad_(True)
    advers_image = input_image.clone().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([advers_image], lr=alpha)

    # TODO this has to be the centroid of the desired label, random noise for now
    target_features = torch.randn(512).to(device)

    for step in range(steps):
        optimiser.zero_grad()

        features = encoder(advers_image)
        loss_embed_desired_label = torch.nn.functional.mse_loss(features, target_features.unsqueeze(0))

        loss_perceptual_image = torch.nn.functional.mse_loss(advers_image, input_image)

        #total loss
        loss = loss_embed_desired_label + (lambda_perceptual*loss_perceptual_image)
        loss.backward()
        optimiser.step()
        
        with torch.no_grad():
            perturbation = torch.clamp(advers_image - input_image, -epsilon, epsilon)
            advers_image.data = torch.clamp(input_image + perturbation, 0, 1)
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
    advers_image = _tensor_to_image(advers_image)

    return advers_image

def _tensor_to_image(t):
    img = t.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img
        
# Load and Prepare Image
def load_image(path, device):
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(path).convert('RGB')
    tensor = preprocessing(image).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.to(device)

# generate_adversarial("dog.jpg", None)