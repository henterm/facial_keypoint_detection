import numpy as np
import torch
import torch.utils.data as data
from glob import glob
import os


class FaceLandmarksDataset(data.Dataset):
  def __init__(self, samples, transforms=None):
    self.samples = samples
    self.transforms = transforms

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    data_path = self.samples[index]

    data = np.load(data_path, allow_pickle=True).item()
    img = data["image"]
    landmarks = data["landmarks"]

    if self.transforms:
      transformed = self.transforms(image=img, keypoints=landmarks)
      img = transformed["image"]
      landmarks = transformed["keypoints"]

    landmarks = torch.tensor(landmarks, dtype=torch.float32).view(-1)

    return img, landmarks
  

def load_processed_data(processed_dir):
    samples = []
    for path in glob(os.path.join(processed_dir, '*.npy')):
        samples.append(path)
    return samples