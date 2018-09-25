"""
Flower classifier
Usage:
    train.py train [options] <dataset_dir>
    train.py -h | --help
    train.py --version

options:
    --save_dir=<name>            Directory to save checkpoint.
                                 [default: ./]
    --gpu                        Run on GPUs. Defaults on CPU.
    --arch=<name>                Pretrained model name.
                                 [default: vgg16]
    --hidden_units=<linears>     Set classifier hidden layers.
                                 [default: 500,200]
    --batch=<size>               Set batch size. 
                                 [default: 32]
    --epochs=<num>               Set epoch number. 
                                 [default: 10]
    --learning_rate=<lr>         Set learning_rate.
                                 [default: 0.001]                                

"train" mode:
    <dataset_dir>                The dataset directory.


"""

from docopt import docopt
import numpy as np


import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

class ClassifierModel(object):
    """classifier model based on a pretrained models
    
    Paramaters
    ----------
    model: str
        the pretrained model name
    n_classes: int
        set number of classes
    linear: list, optional
        List of dimensions of stacked hidden layers. Defaults to [500,]
    dropout: float
        dropout rate. Defaults to 0.5
    """
    def __init__(self, model_name, n_classes=102,
                linear=[500,], dropout=0.5, gpu=False):
        super(ClassifierModel, self).__init__()
        
        self.model_name = model_name
        self.model = getattr(models, model_name)(pretrained=True)
        self.n_classes= n_classes
        self.linear = linear
        self.dropout = dropout
        self.gpu = gpu
        
        # freeze parameters in feature extraction part
        for param in self.model.parameters():
            param.requires_grad = False
        
        # classifier layers
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout)
        layers_=[]
        input_dim = self.model.classifier[0].in_features
        for hidden_dim in linear:
            layer = nn.Linear(input_dim, hidden_dim, bias=True)
            layers_.append(layer)
            layers_.append(relu)
            layers_.append(dropout)
            input_dim = hidden_dim
        final_layer_ = nn.Linear(input_dim, self.n_classes)
        output_ = nn.LogSoftmax(dim=1)

        self.model.classifier = nn.Sequential(*layers_, final_layer_, output_) 
        if gpu:
            self.model = self.model.cuda()
        
    def __call__(self, X):
        return self.model.forward(X)


class AverageMeter(object):
    """Computes and stores the average and current value
    url: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    url: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def checkpoint(trainer,path):
    """ model checkpoint
    TODO: move it to Trainer
    """
    
    trainer.classifier.model.class_to_idx = trainer.dataloaders['train'].dataset.class_to_idx
    checkpoint = {
        'class_to_idx': trainer.classifier.model.class_to_idx,
        'gpu': trainer.gpu,
        'model_name': trainer.classifier.model_name,
        'classifier': trainer.classifier.model.classifier,
        'num_epochs': trainer.num_epochs,
        'state_dict': trainer.classifier.model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    return

class Trainer(object):
    """Train model
    
    Paramaters
    ----------
    dataloaders: dict
        dataloaders for train, valid, test
    classifier: ClassifierModel
        Classifier Model
    criterion: pytorch loss 
        loss function 
    optimizer: pytorch optimizer
        optimizer
    num_epochs:
        epoch number
    gpu: str
        use gpu or not
    
    """
    def __init__(self, dataloaders, classifier, 
                 criterion, optimizer, num_epochs=10, gpu=False, print_freq=50):
        self.dataloaders = dataloaders
        self.classifier = classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.gpu = gpu
        self.count = 0
        self.print_freq = print_freq
        if self.gpu:
            self.criterion = self.criterion.cuda()
            self.classifier.model.cuda()
            
    def train(self):
        self.classifier.model.train()
        for e in range(self.num_epochs):
            for X, y in iter(self.dataloaders['train']):
                self.count += 1
                if self.gpu:
                    X = X.cuda()
                    y = y.cuda()
                inputs = Variable(X)
                ref = Variable(y)

                optimizer.zero_grad()
                pred = self.classifier(inputs)
                loss = criterion(pred, ref)
                loss.backward()
                optimizer.step()
                if self.count % self.print_freq == 0:
                    v_loss, v_accuracy = self.validate()
                    print('Epoch: {e}/{num_epochs}\t'
                         'Validation loss: {v_loss:.4f}\t'
                         'Validation accuracy: {v_accuracy:.4f}%'.format(e=e+1,num_epochs=self.num_epochs,
                                                                    v_loss=v_loss,v_accuracy=v_accuracy))

    def validate(self, subset='valid'): 
        self.classifier.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloaders[subset]):
                if self.gpu:
                    X = X.cuda()
                    y = y.cuda()

                inputs = Variable(X)
                ref = Variable(y)
                pred = self.classifier(inputs)
                
                # accuarcy and losses
                prec1, prec5 = accuracy(pred, ref, topk=(1, 5))
                loss = criterion(pred, ref)
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))
                
        return losses.avg, top1.avg

def load_checkpoint(path):
    ''''''
    checkpoint = torch.load(path)
    model = getattr(models, checkpoint['model_name'])(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Flower classifier')

    data_dir = arguments['<dataset_dir>']
    save_dir= arguments['--save_dir']
    gpu = arguments['--gpu']
    model_name = arguments['--arch']
    batch_size = int(arguments['--batch'])
    num_epochs = int(arguments['--epochs'])
    linear = arguments['--hidden_units'].split(',')
    linear = [int(layer) for layer in linear]
    lr = float(arguments['--learning_rate'])

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    # transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test']),
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { s: torch.utils.data.DataLoader(image_datasets[s], batch_size=batch_size, shuffle=True) 
                   for s in ['train','valid','test']}

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    n_classes = len(cat_to_name)

    if arguments['train']:
        classifier = ClassifierModel(model_name,n_classes=n_classes, linear=linear,gpu=gpu)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(classifier.model.classifier.parameters(), lr=lr)
        trainer = Trainer(dataloaders,classifier,criterion,optimizer,num_epochs=num_epochs, gpu=gpu)
        trainer.train()
        checkpoint(trainer,save_dir+'checkpoint_commands.pt')
        t_loss, t_accuracy = trainer.validate('test')
        print('Test loss: {t_loss:.4f}\t'
              'Test accuracy: {t_accuracy:.4f}%'.format(t_loss=t_loss,t_accuracy=t_accuracy))
    