import torch

def data_stats(
    train_data):
  train_set_data = train_data.data
  print((train_set_data.dtype))

  # CONVERT TO NUMPY and apply transform
  train_set_data = train_data.transform(train_set_data.numpy())

  print(f'Shape of numpy data = {train_set_data.shape}')
  print(f'Shape of tensor data = {train_set_data.shape}')
  print(f'Min of data = {torch.min(train_set_data)}')
  print(f'Max of data = {torch.max(train_set_data)}')
  print(f'Mean of data = {torch.mean(train_set_data)}')
  print(f'Std of data = {torch.std(train_set_data)}')
  print(f'Var of data = {torch.var(train_set_data)}')
