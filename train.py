import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.dataset import FaceLandmarksDataset, load_processed_data
from data.augmentations import get_train_transform, get_val_transform
from models.model import LandmarkModel
from utils.losses import AdaptiveWingLoss


def train(samples, config, num_workers=2):
  train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

  train_dataset = FaceLandmarksDataset(train_samples)
  test_dataset = FaceLandmarksDataset(test_samples)

  train_loader = DataLoader(
      train_dataset,
      batch_size=32,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
      drop_last=True
  )

  test_loader = DataLoader(
      test_dataset,
      batch_size=32,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = LandmarkModel(backbone_name=config['model']['backbone']).to(device)
  optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
  criterion = AdaptiveWingLoss()
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      factor=0.5,
      patience=5,
      verbose=True
  )

  best_loss = float('inf')

  for epoch in range(60):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
      images, targets = batch
      images = images.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, targets)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for batch in test_loader:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        test_loss += criterion(outputs, targets).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    if avg_test_loss < best_loss:
      best_loss = avg_test_loss
      torch.save(model.state_dict(), 'best_model.pth')

    scheduler.step(avg_test_loss)
    print(f"Epoch {epoch+1} | train loss: {avg_train_loss:.4f} | val loss: {avg_test_loss:.4f}")

  return model

  