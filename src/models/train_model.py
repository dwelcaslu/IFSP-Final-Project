import copy
import time

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChestXrayDataSet(Dataset):
    def __init__(self, dataframe, split, round_number, all_rounds=False):
        self.split = split
        self.round_number = round_number
        dataframe['target'] = dataframe["class label"].apply(lambda x: 0 if x == 'No Finding' else 1)
        if all_rounds:
            self.dataframe = dataframe[(dataframe['split'] == split)].reset_index(drop=True)
        else:
            self.dataframe = dataframe[((dataframe['split'] == split)
                                        & (dataframe['round_number'] <= round_number))].reset_index(drop=True)
        self.image_paths = self.dataframe["img_filepath"].values
        self.targets = torch.FloatTensor(self.dataframe['target'].values)
        self.CLASSES_LABELS = ['Healthy', 'Sick']
        self.TARGET_DICT = {self.CLASSES_LABELS[i]: i for i in range(len(self.CLASSES_LABELS))}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
        ])
        image = preprocess(image)

        return image, self.targets[index]


def get_dataloaders(df, round_number, test_all_rounds=False):

    train_set = ChestXrayDataSet(df, 'train', round_number)
    valid_set = ChestXrayDataSet(df, 'valid', round_number)
    test_set = ChestXrayDataSet(df, 'test', round_number, all_rounds=test_all_rounds)
    print('train_set size:', train_set.__len__())
    print('valid_set size:', valid_set.__len__())
    print('test_set size:', test_set.__len__())
    print('total:', train_set.__len__() + valid_set.__len__() + test_set.__len__())
    db_sizes = {'train': train_set.__len__(), 'valid': valid_set.__len__(),
                'test': test_set.__len__()}

    # Setting the dataloaders:
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_set, batch_size=1, shuffle=False)
    dataloaders = {'train': train_dl, 'valid': valid_dl, 'test': test_dl}

    return dataloaders, db_sizes


def train_model(model, dataloaders, device,
                criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model_hist = []

    for epoch in range(num_epochs):
        since_epoch = time.time()
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        epoch_perf = {'epoch': epoch}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_samples = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                n_samples += len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    preds = outputs.round()
                    loss = criterion(outputs, labels.reshape(len(labels), 1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.reshape(len(preds)) == labels.data)

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            epoch_perf[f'{phase}_loss'] = epoch_loss
            epoch_perf[f'{phase}_acc'] = epoch_acc
            if phase == 'train':
                scheduler.step()
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        model_hist.append(epoch_perf)
        time_elapsed_epoch = time.time() - since_epoch
        print(f'{time_elapsed_epoch // 60:.0f}m {time_elapsed_epoch % 60:.0f}s/epoch')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(model_hist)

def set_and_train_model(model, dataloaders, num_epochs=10):
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # RMSprop, Adam
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    model, model_hist = train_model(model, dataloaders, DEVICE, criterion, optimizer,
                                    exp_lr_scheduler, num_epochs=num_epochs)

    return model, model_hist
