import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

import json
import math


def main():
    pass


    arguments = get_args()

    data_dir = arguments.data_directory
    save_dir = arguments.save_dir
    arch = arguments.arch
    learning_rate = arguments.learning_rate
    epochs = arguments.epochs
    hidden_layers = arguments.hidden_units
    gpu = arguments.gpu
    outputs = arguments.outputs

    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    dataloaders, image_datasets  = load_datasets(train_dir, valid_dir, test_dir)

    model = build_model(nn_type=arch, outputs = outputs, hidden_layers = hidden_layers)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print('Training start')
    train(model, device, criterion, optimizer, dataloaders['train'], dataloaders['test'], epochs=epochs, print_every = 20)
    print('Training finish')

    save_checkpoint(model, arch, hidden_layers,  image_datasets, model_name = save_dir)

def load_datasets(train_dir, valid_dir, test_dir):

    noramlize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         noramlize]),
        'valid': transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         noramlize]),
        'test': transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         noramlize])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test':  torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    }

    return dataloaders, image_datasets


def get_in_features(model):

    for item in  model.classifier:
        if type(item) == torch.nn.modules.linear.Linear:
            return item.in_features


def build_model(nn_type='vgg16', outputs = 102, hidden_layers = [1000]):

    model = getattr(models, nn_type)(pretrained=True)
    print(model)

    in_features = get_in_features(model)

    hidden_layers = [in_features] + hidden_layers

    j = len(hidden_layers) - 1
    i = 0

    #hidden layers go here
    layers = []
    while j > i:
        layers.append((f'fc{i}', nn.Linear(hidden_layers[i], hidden_layers[i+1])))
        layers.append((f'relu{i}', nn.ReLU()))
        layers.append((f'dropout{i}', nn.Dropout()))
        i+=1

    #final layer of the network
    layers.append(('fc_last', nn.Linear(hidden_layers[-1], outputs)))
    layers.append(('output', nn.LogSoftmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(layers))

    # Replace classifier
    model.classifier = classifier

    return model


def validate(model, device, criterion, testing_set):

    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testing_set:

            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            batch_loss = criterion(output, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    return test_loss/len(testing_set), accuracy/len(testing_set)


def train(model, device, criterion, optimizer, traning_set, testing_set, epochs=5, print_every = 20):

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in traning_set:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss, accuracy = validate(model, device, criterion, testing_set)

                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss:.3f}.. "
                  f"Test accuracy: {accuracy:.3f}")

                running_loss = 0



def save_checkpoint(model, arch, hidden_layers,  image_datasets, model_name = "checkpoint.pt"):

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'arch' : arch,
        'class_to_idx' : model.class_to_idx,
        'state_dict' : model.state_dict(),
        'hidden_layers' : hidden_layers
    }

    torch.save(checkpoint, model_name)
    print('saving checkpoint as: {}'.format(model_name))


def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_directory", type=str, default = 'flowers', help="data directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pt" ,help="directory where to save trained model and hyperparameters")
    parser.add_argument("--arch", type=str, default="vgg16", help="pre-trained model: vgg16, alexnet")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs to train model")
    parser.add_argument("--hidden_units", type=list, default=[768, 384], help="list of hidden layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gpu", type=bool, default=True, help="use GPU or CPU to train model: True = GPU, False = CPU")
    parser.add_argument("--outputs", type=int, default=102, help="enter output size")

    return parser.parse_args()

if __name__ == "__main__":
    main()
