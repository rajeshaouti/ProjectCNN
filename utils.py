"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path

import matplotlib.pyplot as plt 

def visualize_input_data(train_set,class_names,rows,cols):

  torch.manual_seed(42)

  fig = plt.figure(figsize=(9, 9))
  rows, cols = rows,cols
  for i in range(1, rows * cols + 1):
      random_idx = torch.randint(0, len(train_set), size=[1]).item()
      img, label = train_set[random_idx]
      fig.add_subplot(rows, cols, i)
      plt.imshow(img.squeeze(), cmap="gray")
      plt.title(class_names[label])
      plt.axis(False);

import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
