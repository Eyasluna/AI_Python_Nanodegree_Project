"""
Flower prediction
Usage:
    predict.py predict [options] <image_path> <checkpoint_path>
    predict.py -h | --help
    predict.py --version

options:
    --gpu                               Run on GPUs. Defaults on CPU.
    --top_k=<k>                         Top k predicited classes
                                        [default: 5]   
    --category_names=<json_path>        The mapping of categories to real names
                                        [default: cat_to_name.json]
    
"predict" mode:
    <image_path>                 The input image path.
    <checkpoint_path>            The trained model path.


"""

from docopt import docopt
import json
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image

def load_checkpoint(path):
    """load checkpoint
    """
    checkpoint = torch.load(path)
    model = getattr(models, checkpoint['model_name'])(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    # resize image
    width, height = image.size
    aspect_ratio = width/height
    n_width, n_height = 0,0
    if width < height:
        n_width = 256  
    else:
        n_height = 256
        
    if n_width==0:
        n_width = int(n_height*aspect_ratio)
    else:
        n_height = int(n_width/aspect_ratio)
        
    image = image.resize((n_width, n_height))
    
    # crop the center of the image
    new_width, new_height = (224, 224)

    left = (n_width - new_width)/2
    top = (n_height - new_height)/2
    right = (n_width + new_width)/2
    bottom = (n_height + new_height)/2

    image = image.crop((left, top, right, bottom))
    
    # convert image color channels to floats 0-1. 
    np_image = np.array(image)
    np_image = np_image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
                       
    # put the color channel to the first dimension
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # reset device to cpu 
    model.cpu()
    model.eval()
    
    # idx to class mapping
    idx_to_class = {i: k for k, i in model.class_to_idx.items()}

    # process image
    image = process_image(Image.open(image_path))
    image = torch.FloatTensor([image])

    if gpu:
        model = model.cuda()
        image = image.cuda()

    idx_to_class = {i: k for k, i in model.class_to_idx.items()}
    
    #prediction
    output = model.forward(Variable(image))
    probas = torch.exp(output).data[0]
    k_proba, k_pred = probas.topk(topk)
    k_classes = [idx_to_class[idx] for idx in k_pred.cpu().numpy()]

    return list(k_proba.cpu().numpy()) , k_classes
    


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Flower prediction')
    
    if arguments['predict']: 
        category_names = arguments['--category_names']
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_k = int(arguments['--top_k'])

        image_path = arguments['<image_path>']
        checkpoint_path = arguments['<checkpoint_path>']
        model = load_checkpoint(checkpoint_path)
        probs, classes = predict(image_path, model, topk = top_k)
        flower_names = [cat_to_name[c] for c in classes]
        print(probs)
        print(classes)
        print(flower_names)