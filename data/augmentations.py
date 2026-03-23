import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
  transform = A.Compose([A.Resize(height=112, width=112),
                         A.Rotate(p=0.5),
                         A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
                         A.GaussianBlur(sigma_limit=(0.5, 2.0), blur_limit=0, p=0.15),
                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ToTensorV2()
                         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
  return transform

def get_val_transform():
  transform = A.Compose([A.Resize(height=112, width=112),
                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ToTensorV2()
                         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
  return transform