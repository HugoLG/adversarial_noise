import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

"""
    Given an input image the function will use a pretrained model to extract features from it.
    Default pretrained model to be used is VGG16.
    Params:
        @img_file can be img object or string img file path
        @is_img_path boolean to define if img_file is object or file path
"""
def extract_features(img_file, is_img_path=False, preprocess=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load image to img variable
    if is_img_path:
        img = img_file
    else:
        img = Image.open(img_file).convert("RGB")

    # 2. Define model and get only the encoding part, not the classifier head
    model = models.vgg16(pretrained=True)
    # encoder = model.features[:23].eval()
    # encoder.to(device)
    encoder = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((7, 7)),
        torch.nn.Flatten()   # yields [1,512]
    ).to(device).eval()
    
    # 3. Create Image preprocessing pipeline for image use pytorch transforms
    # Use 224x224 for VGG16
    # TODO consider moving this to a function, it is used in multiple places of the code
    preprocessing = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. pass input to encoder/img classifier model
    input = preprocessing(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(input)
    features = features.view(features.size(0), -1).squeeze(0)

    return features.cpu()
