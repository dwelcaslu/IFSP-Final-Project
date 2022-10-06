import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow(inp, title=None, figsize=(8, 6)):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def plot_samples(dataloader, title='images samples', figsize=(12, 10)):

    # Get a batch of training data
    inputs, classes = next(iter(dataloader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=title, figsize=figsize)


def plot_model_hist(historic, title='', figsize=(15, 5)):

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.title(title+' loss')
    plt.plot(historic['train_loss'], label='train loss')
    plt.plot(historic['valid_loss'], label='valid. loss')
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.title(title+' acc')
    plt.plot(historic['train_acc'], label='train acc')
    plt.plot(historic['valid_acc'], label='valid. acc')
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()
