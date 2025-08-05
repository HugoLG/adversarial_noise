import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
from pathlib import Path
from feature_extraction import *

def process_label_folder(label_folder_path):
    label_folder = Path(label_folder_path)
    images_folder = label_folder / "images"

    if not images_folder.exists():
        print(f"Images folder not found in {label_folder}")
        return None

    image_extensions = {".jpg", ".jpeg"}
    image_files = [f for f in images_folder.iterdir() if f.suffix.lower() in image_extensions]

    if len(image_files) == 0:
        print(f"No images found in {images_folder}")
        return None

    # Extract features for all images
    features_list = []
    for image_file in image_files:
        try:
            # features = extractor.extract_features(image_file)
            features = extract_features(image_file)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    if not features_list:
        return None

    features_array = np.array(features_list)
    centroid = np.mean(features_array, axis=0)

    label_name = label_folder.name
    pickle_path = label_folder / f"{label_name}_centroid.pkl"

    with open(pickle_path, "wb") as f:
        pickle.dump(centroid, f)

    print(f"Saved centroid for {label_name} to {pickle_path}")
    print(f"Centroid shape: {centroid.shape}")

    return centroid

def extract_typical_features(parent_folder_path):
    parent_folder = Path(parent_folder_path)

    label_folders = [f for f in parent_folder.iterdir() if f.is_dir()]

    if not label_folders:
        print(f"No label folders found in {parent_folder}")
        return None

    print(f"Found {len(label_folders)} label folders")

    centroids = {}
    for label_folder in label_folders:
        centroid = process_label_folder(label_folder)
        if centroid is not None:
            centroids[label_folder.name] = centroid

    print(f"Processing complete. {len(centroids)} centroids extracted")
    return centroids
