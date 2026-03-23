import os
import cv2
import dlib
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models.model import LandmarkModel
from data.augmentations import get_val_transform


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmark points on image."""
    image = image.copy()
    h, w = image.shape[:2]
    for (x, y) in landmarks:
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(image, (cx, cy), radius, color, -1)
    return image


def predict(model, image, detector, transform, device):
    """Run face detection and landmark prediction on a single image."""
    dets = detector(image, 1)
    if not dets:
        return None, None

    bbox = dets[0]
    x1, y1, x2, y2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
    w, h = x2 - x1, y2 - y1

    margin = 0.3
    x1 = max(0, int(x1 - margin * w))
    y1 = max(0, int(y1 - margin * h))
    x2 = min(image.shape[1], int(x2 + margin * w))
    y2 = min(image.shape[0], int(y2 + margin * h))

    face_crop = image[y1:y2, x1:x2]
    transformed = transform(image=face_crop)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    landmarks = output.cpu().numpy().reshape(-1, 2)
    return landmarks, (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='convnext_tiny')
    parser.add_argument('--output_path', type=str, default='result.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LandmarkModel(backbone_name=args.backbone).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    detector = dlib.get_frontal_face_detector()
    transform = get_val_transform()

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks, bbox = predict(model, image, detector, transform, device)

    if landmarks is None:
        print("No face detected.")
        return

    result = draw_landmarks(image, landmarks)

    plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Result saved to {args.output_path}")


if __name__ == '__main__':
    main()