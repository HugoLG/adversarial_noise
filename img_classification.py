import torch
import torch.nn as nn
from torchvision import models, transforms
import urllib.request
from PIL import Image


def classify_image(img_file, is_img_path=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load image to img variable
    if is_img_path:
        img = img_file
    else:
        img = Image.open(img_file).convert("RGB")
    
    # 2. Get img classifier from torch models library
    model = models.vgg16(pretrained=True)
    model.eval()
    model = model.to(device)

    # 3. Create Image preprocessing pipeline for image use pytorch transforms
    # Use 224x224 for VGG16
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 4. preprocess img and create the batch format to pass it to model
    tensor = preprocessing(img)
    input_batch = tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    # 5. pass it to model and get outputs
    with torch.no_grad():
        outputs = model(input_batch)

    # 6. the outputs have the logits but we also need the string class label
    # we can attempt get the imagenet classes from the following link
    classes = get_imagenet_classes()

    # process output and return label, confidence, probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_i = probabilities.argmax().item()
    label = classes[top_i] if classes is not None else "Labels not available."
    confidence = probabilities[top_i].item()

    return {"label":label, "confidence":confidence, "probabilities":probabilities}


def get_imagenet_classes():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            classes = [line.strip().decode("utf-8") for line in resp.readlines()]
    except Exception as e:
        print("Failed to download label mapping:", e)
        classes = None
    return classes

