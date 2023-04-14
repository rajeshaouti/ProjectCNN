import matplotlib.pyplot as plt
import numpy as np
import torch
     
def visualize_input_data(train_set,class_names,rows,cols):

  torch.manual_seed(42)

  fig = plt.figure(figsize=(15, 15))
  rows, cols = rows,cols
  for i in range(1, rows * cols + 1):
      random_idx = torch.randint(0, len(train_set), size=[1]).item()
      img, label = train_set[random_idx]
      fig.add_subplot(rows, cols, i)
      img = img.numpy()
      plt.imshow(img.transpose(1,2,0))
      plt.title(class_names[label])
      plt.axis(True);

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 3.5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def plot_misclassified_images(model,num_of_images,test_dataloader,classes,device):

  model.eval()

  figure = plt.figure(figsize=(20, 20))
  # num_of_images = 5
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

  return misclass_img_list
