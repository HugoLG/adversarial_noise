import torch
import numpy as np
from feature_extraction import *
from torchvision import models, transforms
from img_classification import *
import pickle

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
"""
This function adds the adversarial noise to the input image
Then tries to optimise the features vector so that it is close to original input image

"""
def generate_adversarial(input_image, desired_label, epsilon=0.03, alpha=0.05, steps=500, lambda_perceptual=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.vgg16(pretrained=True).to(device).eval()
    classifier = model
    encoder = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(start_dim=1, end_dim=2)   # yields [1,512]
    ).to(device).eval()

    input_image = load_image(input_image, device)
    input_image = input_image.requires_grad_(True)
    advers_image = input_image.clone().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([advers_image], lr=alpha)

    #get centroid for desired label
    # TODO missing error handling
    centroid_path = f"{desired_label}/{desired_label}_centroid.pkl"
    print(f"centroid path to be open: {centroid_path}")
    with open(centroid_path, "rb") as f:
        centroid_np = pickle.load(f)
    target_features = torch.from_numpy(centroid_np).unsqueeze(0).to(device).float()

    # get class index
    classes = get_imagenet_classes()
    target_index = classes.index(desired_label)
    print(f"selected class: {classes[target_index]}, index: {target_index}")

    for step in range(steps):
        optimiser.zero_grad()

        features = encoder(advers_image)
        # loss_embed_desired_label = torch.nn.functional.mse_loss(features, target_features.unsqueeze(0))
        loss_embed_desired_label = torch.nn.functional.mse_loss(features, target_features)

        logits = classifier(advers_image)
        loss_classes = torch.nn.functional.cross_entropy(logits, torch.tensor([target_index], device=device))

        loss_perceptual_image = torch.nn.functional.mse_loss(advers_image, input_image)

        #total loss
        loss = 10*loss_embed_desired_label + (5*lambda_perceptual*loss_classes) + (lambda_perceptual*loss_perceptual_image)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            perturbation = torch.clamp(advers_image - input_image, -epsilon, epsilon)
            advers_image.data = torch.clamp(input_image + perturbation, 0, 1)

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    x_unnorm = unnormalize_tensor(advers_image.squeeze(0).cpu())
    pil = tensor_to_pil(x_unnorm) 
    advers_image = _tensor_to_image(advers_image)

    return pil

def tensor_to_pil(x_tensor):
    """
    x_tensor: [3,H,W] un-normalized, in [0,1] range
    """
    x_tensor = torch.clamp(x_tensor, 0.0, 1.0)
    to_pil = transforms.ToPILImage()
    return to_pil(x_tensor)

def unnormalize_tensor(x_norm):
    """
    Invert the ImageNet normalization so we can convert back to PIL.
    x_norm is shape [3,H,W], float in normalized space.
    """
    inv_mean = [-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std  = [1/s  for s    in IMAGENET_STD]
    inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)
    return inv_norm(x_norm)

def _tensor_to_image(t):
    img = t.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img

# Load and Prepare Image
def load_image(path, device):
        # transforms.CenterCrop(224),
    preprocessing = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(path).convert('RGB')
    tensor = preprocessing(image).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.to(device)

"""
Wrapper for generate_adversarial function.
This gives the option to save the output adversarial image to @param output_filepath
It also return the adversarial image, same as generate_adversarial.
"""
def generate_and_save_adversarial(input_image, desired_label, output_filepath="output_adversarial_image.jpg", epsilon=0.03, alpha=0.05, steps=500, lambda_perceptual=1.0):
    advers_image = generate_adversarial(input_image, desired_label, epsilon=epsilon, alpha=alpha, steps=steps, lambda_perceptual=lambda_perceptual)
    advers_image.save(output_filepath)

    return advers_image
