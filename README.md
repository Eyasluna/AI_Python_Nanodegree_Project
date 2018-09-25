# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Environment
Please install [`docopt`](http://docopt.org/) for command arguments parsing.

## Training
```bash
$ python train.py -h

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

```
```bash
$ python train.py train --gpu --hidden_units=512 --epochs=2 --learning_rate=0.001 --save_dir=~/ flowers
Epoch: 1/5	Validation loss: 2.7958	Validation accuracy: 40.8313%
Epoch: 1/5	Validation loss: 1.2900	Validation accuracy: 63.5697%
...
Epoch: 5/5	Validation loss: 0.5100	Validation accuracy: 85.8191%
Epoch: 5/5	Validation loss: 0.4535	Validation accuracy: 88.1418%
Test loss: 0.5898	Test accuracy: 84.1270%

```
## Prediction
```bash
$ python predict.py predict -h
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
```

```bash
$ python predict.py predict ./flowers/test/28/image_05230.jpg checkpoint_commands.pt --gpu --top_k=3
[0.7435395, 0.1538978, 0.046431974]
['28', '9', '45']
['stemless gentian', 'monkshood', 'bolero deep blue']

```
