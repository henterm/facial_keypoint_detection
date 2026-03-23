import os
import cv2
import dlib
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm


def load_pts(pts_path):
  try:
    with open(pts_path) as f:
      lines = [line.strip() for line in f.readlines()]

    points = []
    start = False
    for line in lines:
      if line == "{":
        start = True
        continue
      if line == "}":
        break
      if start and line:
        x, y = map(float, line.split())
        points.append((x, y))

    return np.array(points) if len(points) == 68 else None
  except:
    return None
  
  
def process_sample(img_path, pts_path, detector):
  try:
    img = cv2.imread(img_path)
    if img is None:
      return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = load_pts(pts_path)

    if landmarks is None:
      return None

    dets = detector(img, 1)
    if not dets:
      return None

    bbox = dets[0]
    x1, y1, x2, y2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
    w, h = x2 - x1, y2 - y1

    margin = 0.3
    x1 = max(0, int(x1 - margin * w))
    y1 = max(0, int(y1 - margin * h))
    x2 = min(img.shape[1], int(x2 + margin * w))
    y2 = min(img.shape[0], int(y2 + margin * h))

    if x1 >= x2 or y1 >= y2:
      return None

    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (112, 112))

    norm_landmarks = []
    for (x, y) in landmarks:
      nx = np.clip((x - x1) / (x2 -  x1), 0.0, 1.0)
      ny = np.clip((y - y1)/ (y2 - y1), 0.0, 1.0)
      norm_landmarks.append([nx, ny])

    norm_landmarks = np.array(norm_landmarks, dtype=np.float32)

    return {
            'image': face_resized,
            'landmarks': norm_landmarks
        }

  except Exception as e:
    print(f"Error in {img_path}: {str(e)}")
    return None
  

def prepare_data(dataset_paths, output_dir='processed'):
    os.makedirs(output_dir, exist_ok=True)
    detector = dlib.get_frontal_face_detector()

    print("Preprocessing data...")
    samples = []

    for dataset_path in dataset_paths:
        img_paths = glob(f"{dataset_path}/**/*.jpg", recursive=True)
        for img_path in tqdm(img_paths, desc=f"Processing {dataset_path}"):
            pts_path = os.path.splitext(img_path)[0] + '.pts'
            if not os.path.exists(pts_path):
                continue

            result = process_sample(img_path, pts_path, detector)
            if result is None:
                continue

            base_name = os.path.basename(img_path)
            save_path = os.path.join(output_dir, base_name.replace('.jpg', '.npy'))
            np.save(save_path, result)
            samples.append(save_path)

    print(f"\nSuccessfully processed {len(samples)} samples")
    return samples