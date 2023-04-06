"""
Contains functionality for creating PyTorch DataLoaders for image classification 
data using the data form pytorch vision transforms datasets

"""
import os

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_set: torchvision.datasets, 
    test_set: torchvision.datasets, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training data and testing data in form of torchvision.datasets and turns
  them into PyTorch DataLoaders.

  Args:
    train_set: Train data fromtorchvision.datasets.
    test_set: Test data fromtorchvision.datasets.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
  """
  
  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_set,batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  test_dataloader = DataLoader(
      test_set,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader
