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


def get_in_features(model):

    for item in  model.classifier:
        if type(item) == torch.nn.modules.linear.Linear:
            return item.in_features


def build_model(nn_type='vgg16', outputs = 102, hidden_layers = [1000]):

    model = getattr(models, nn_type)(pretrained=True)
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


def load_checkpoint(path):
    checkpoint = torch.load(path)

    arch = checkpoint['arch']
    out_features  = len(checkpoint['class_to_idx'])
    hidden_layers = checkpoint['hidden_layers']

    model = build_model(arch, out_features, hidden_layers)
    model.load_state_dict(checkpoint['state_dict'])

    print(model.classifier)

    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    pil_image = Image.open(image_path)

    # resize
    if pil_image.size[1] < pil_image.size[0]:
        pil_image.thumbnail((255, math.pow(255, 2)))
    else:
        pil_image.thumbnail((math.pow(255, 2), 255))

    # crop
    left = (pil_image.width-224)/2
    bottom = (pil_image.height-224)/2
    right = left + 224
    top = bottom + 224

    pil_image = pil_image.crop((left, bottom, right, top))

    # Turn into np_array
    np_image = np.array(pil_image)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    tensor_img = torch.FloatTensor([process_image(image_path)])

    tensor_img = tensor_img.to(device)
    result = model(tensor_img).topk(topk)


    probs = torch.exp(result[0].data).cpu().numpy()[0]
    indeces = result[1].data.cpu().numpy()[0]

    return probs, indeces

def main():

    arguments = get_args()

    image_path = arguments.image_path
    checkpoint = arguments.checkpoint
    category_names = arguments.category_names
    top_k = arguments.top_k
    gpu = arguments.gpu

    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    loaded_model = load_checkpoint(checkpoint)
    #criterion = nn.NLLLoss()
    #optimizer = optim.Adam(loaded_model.classifier.parameters(), lr=0.001)
    test_image_path  = 'flowers/test' + image_path

    test_image_name = cat_to_name[test_image_path.split('/')[-2]]

    probs, idxs = predict(test_image_path, loaded_model, device, topk=top_k)

    idx_to_class = {v: k for k, v in loaded_model.class_to_idx.items()}

    names = [cat_to_name[idx_to_class[x]] for x in idxs]
    classes = [idx_to_class[x] for x in idxs]

    print('\nTOP Result')
    print('target: ', test_image_name)
    print(names)
    print(probs)
    print(idxs)
    print(classes)

    print('\nAbsolute Result')
    print('target: ', test_image_name)
    print(names[0])
    print(probs[0])
    print(idxs[0])
    print(classes[0])


def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()

    # change to match predict.py instructions
    parser.add_argument("--image_path", type=str, default="/59/image_05020.jpg", help="Image target. making a prediciton over it")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="checkpoint for trained model")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Categories to real names")
    parser.add_argument("--top_k", type=int, default=3, help="Top classes")
    parser.add_argument("--gpu", type=bool, default=True, help="use GPU or CPU to train model: True = GPU, False = CPU")


    return parser.parse_args()


if __name__ == "__main__":
    main()
