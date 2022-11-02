import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torchvision


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenseNet121(nn.Module):
    def __init__(self, n_classes=1, finetune=False):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = finetune
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model(n_classes=1, finetune=True, model_name=None, base_path='../../models/'):
    model = DenseNet121(n_classes=n_classes, finetune=finetune)
    if model_name:
        model_path = f'{base_path}{model_name}.pth'
        try:
            model.load_state_dict(torch.load(model_path))
            print(f'\n{model_name}.pth loaded successfully.')
        except FileNotFoundError:
            print(f'\nFile Not Found: {model_path}.\nModel will start with default initialization.')
    model = model.to(DEVICE)
    print(f'Cuda available: {torch.cuda.is_available()}. Model sent to device: {DEVICE}.')
    return model


def evaluate_model(model, dataloaders, split='test'):
    model.eval()
    y_pred = np.array([[-1.]])
    y_true = np.array([[-1.]])
    for _, (inputs, labels) in enumerate(dataloaders[split]):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        # _, preds = torch.max(outputs, 1)
        preds = outputs.round()
        y_pred = np.vstack((y_pred, preds.cpu().detach().numpy()))      
        y_true = np.vstack((y_true, labels.cpu().detach().numpy()))      

    y_pred = y_pred[1:, :]; y_true = y_true[1:, :]
    print(f'\nClassification report for {split} set:')
    print(classification_report(y_true, y_pred))
    model_stats = {'accuracy_score': accuracy_score(y_true, y_pred),
                   'balanced_accuracy_score': balanced_accuracy_score(y_true, y_pred),
                   'recall_score': recall_score(y_true, y_pred),
                   'precision_score': precision_score(y_true, y_pred),
                   'f1_score': f1_score(y_true, y_pred),
                   'confusion_matrix': confusion_matrix(y_true, y_pred),
                #    'classification_report': classification_report(y_true, y_pred),
                   }
    return model_stats


def save_model(model, model_name, base_path='../../models/'):
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    model_path = f'{base_path}{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'{model_name} saved at {model_path}')
