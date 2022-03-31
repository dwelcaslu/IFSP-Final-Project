import syft as sy
import torch
import torchvision
from collections import OrderedDict
from tools.pytorch_fl import ConvNet
import os
import urllib
import tarfile

# method copied from https://github.com/OpenMined/PySyft/blob/dev/packages/syft/src/syft/util.py
from pathlib import Path
def get_root_data_path() -> Path:
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"

    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def create_and_upload_mnist_datasets_on_owner(tags, descriptions, duet_server):
    # download the mnist dataset from torchvision
    print("Downloading the datasets ...")
    train_set = torchvision.datasets.MNIST(str(get_root_data_path()), train=True, download=True)
    eval_set = torchvision.datasets.MNIST(str(get_root_data_path()), train=False, download=True)

    # # download the MNIST dataset
    # if not os.path.exists("MNIST.tar.gz"):
    #     urllib.urlretrieve ("http://www.di.ens.fr/~lelarge/MNIST.tar.gz", "MNIST.tar.gz")
    #     tar = tarfile.open("MNIST.tar.gz", "r:gz")
    #     tar.extractall()
    #     tar.close()
    
    # # create the pytorch dataset
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # train_set = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    # eval_set = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)

    # upload the training/evaluation sets length to the duet server
    print("Uploading the dataset sizes to the duet server ...")
    train_set_length = torch.IntTensor([len(train_set)])
    train_set_length = train_set_length.tag(tags['train'])
    train_set_length = train_set_length.describe(descriptions['train'])
    train_set_length_ptr = train_set_length.send(duet_server, searchable=True)
    
    eval_set_length = torch.IntTensor([len(eval_set)])
    eval_set_length = eval_set_length.tag(tags['eval'])
    eval_set_length = eval_set_length.describe(descriptions['eval'])
    eval_set_length_ptr = eval_set_length.send(duet_server, searchable=True)

def get_mnist_dataset_loader(torch_ref, torchvision_ref, batch_size, is_local=True, duet_server=None, train_set=True):
    transform_1 = torchvision_ref.transforms.ToTensor()
    transform_2 = torchvision_ref.transforms.Normalize(0.1307, 0.3081)
    
    if is_local:
        transforms = torchvision_ref.transforms.Compose([transform_1, transform_2])
    else:
        remote_list = duet_server.python.List()  # create a remote list to add the transforms to
        remote_list.append(transform_1)
        remote_list.append(transform_2)
        transforms = torchvision_ref.transforms.Compose(remote_list)

    kwargs = {"batch_size": batch_size}

    dataset = torchvision_ref.datasets.MNIST(str(get_root_data_path()), train=train_set, download=True, transform=transforms)
    loader = torch_ref.utils.data.DataLoader(dataset,**kwargs)

    return dataset, loader

def verify_remote_cuda(remote_torch, args, model):
    # lets ask to see if our Data Owner has CUDA
    has_cuda = False
    has_cuda_ptr = remote_torch.cuda.is_available()
    has_cuda = bool(has_cuda_ptr.get(
        request_block=True,
        reason="To run test and inference locally",
        timeout_secs=5,  # change to something slower
    ))

    use_cuda = not args["no_cuda"] and has_cuda
    # now we can set the seed
    remote_torch.manual_seed(args["seed"])

    device = remote_torch.device("cuda" if use_cuda else "cpu")
    print(f"  Data Owner device is {device.type.get()}!")

    # if we have CUDA lets send our model to the GPU
    if has_cuda:
        model.cuda(device)
    else:
        model.cpu()

def combine_remote_convnet_models(remote_models, train_set_sizes):
    weights = OrderedDict()

    # compute the weighted average of the model weights/biases
    print("> Downloading remote models and averaging them")
    for i in range(len(remote_models)):
        print('\n------------------------------------')
        print(f'> Model #{i+1}')
        print('------------------------------------\n')
        # save the remote trained model
        local_model_updates = remote_models[i].get(
            request_block=True,
            reason="Save the trained model",
            timeout_secs=5
        ).state_dict()

        if len(weights) == 0:
            for layer in ['conv1', 'conv2', 'fc1', 'fc2']:
                weights[f'{layer}.weight'] = local_model_updates[f'{layer}.weight']*train_set_sizes[i]
                weights[f'{layer}.bias'] = local_model_updates[f'{layer}.bias']*train_set_sizes[i]
        else:
            for layer in ['conv1', 'conv2', 'fc1', 'fc2']:
                weights[f'{layer}.weight'] += local_model_updates[f'{layer}.weight']*train_set_sizes[i]
                weights[f'{layer}.bias'] += local_model_updates[f'{layer}.bias']*train_set_sizes[i]

    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:
        weights[f'{layer}.weight'] = weights[f'{layer}.weight'] / sum(train_set_sizes)
        weights[f'{layer}.bias'] = weights[f'{layer}.bias'] / sum(train_set_sizes)

    # create an empty model and load the combined weights to it
    print('\n------------------------------------')
    print(f'> Creating the new combined model')
    print('------------------------------------\n')
    combined_model = ConvNet(torch)
    combined_model.load_state_dict(weights)

    return combined_model