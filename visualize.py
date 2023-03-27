import matplotlib.pyplot as plt
import numpy as np
import torch

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

def plot_misclassified_images(model,num_of_images,test_dataloader,classes):

  model.eval()

  figure = plt.figure(figsize=(20, 20))
  # num_of_images = 25
  index = 1

  misclass_img_list = []
  untrans_img=[]

  with torch.no_grad():
      for data, target in test_dataloader:
          data, target = data.to(
              device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)
          act = target.view_as(pred)
          # since most of the bool vec is true (good problem to have) and switch (flip) the true to false and vice versa
          bool_vec = ~pred.eq(act)

          # now extract the index number from the tensor which has 'true'
          idx = list(
              np.where(bool_vec.cpu().numpy())[0])

          if idx:  # if not a blank list
              idx_list = idx
              # print(data[idx_list[0]].shape)
              if index < num_of_images+1:
                  plt.subplot(5, 5, index)
                  plt.axis('off')
                  title = 'act/pred : ' + \
                      str(classes[target[idx[0]].cpu().item(
                      )]) + '/' + str(classes[pred[idx[0]].cpu().item()])
                  # prints the 1st index of each batch.
              
                  img = data[idx[0]].cpu()
                  untrans_img.append(img)
                  image = plt.imshow(img.permute(1,2,0))
                  misclass_img_list.append(image)
                                    
                  plt.title(title)
                  index +=  1
