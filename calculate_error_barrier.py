import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models.utils import get_model
from data.utils import get_loader
from optim.utils import get_optim
from utils import Logger, AverageMeter
from tqdm import tqdm

import argparse
import os
import random
import copy
import datetime

import matplotlib
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()

# Datasets`
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--imagenet_root', default='./imagenet', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--pretrained1', type=str, default=None)
parser.add_argument('--pretrained2', type=str, default=None)

parser.add_argument('--name', help='name of the experiment', default='None')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def main():
    # Model
    model = get_model(args)
    model = model.cuda()

    initial_model = copy.deepcopy(model)

    trainloader, testloader = get_loader(args)

    criterion = nn.CrossEntropyLoss()

    calculate_error_barrier(args.pretrained1, args.pretrained2, model, initial_model, trainloader, testloader, criterion, args)

    plt.xlabel('Interpolation Ratio')
    plt.ylabel("Error (%)")
    plt.tight_layout()
    
    os.makedirs(f'plots/{args.dataset}/{args.arch}', exist_ok=True)
    plt.savefig(f'plots/{args.dataset}/{args.arch}/{args.name}.png')

def train(trainloader, model):
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.alpha = 0

    losses = AverageMeter()
    acc = AverageMeter()

    pbar = tqdm(trainloader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            pbar.set_postfix(loss=losses.avg, acc=acc.avg)
    

def test(testloader, model, criterion):
    model.eval()

    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            correct = (outputs.max(1)[1]).eq(targets).sum()
            losses.update(loss.item(), inputs.size(0))
            acc.update(correct.item()/inputs.size(0)*100, inputs.size(0))

    return (losses.avg, acc.avg)

def calculate_error_barrier(pretrained1, pretrained2, model, initial_model, trainloader, testloader, criterion, args):
    pretrained = torch.load(pretrained1)['state_dict']
    model.load_state_dict(pretrained)

    model_tmp = copy.deepcopy(model)
    model_tmp.load_state_dict(torch.load(pretrained2)['state_dict'])

    accs = []
    alphas = []
    num_interpolation = 10
    for alpha in range(0,num_interpolation+1):
        interpolated_model = copy.deepcopy(initial_model)
        alpha=alpha/num_interpolation

        for m1, m2, m3 in zip(model.modules(), model_tmp.modules(), interpolated_model.modules()):
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear) or isinstance(m1, nn.BatchNorm2d):
                if hasattr(m1, 'fixed_m'):
                    assert (m1.fixed_m != m2.fixed_m).sum() == 0, 'Two models have differen pruning masks'
                    m3.fixed_m.data = copy.deepcopy(m1.fixed_m.data)

                w = copy.deepcopy(alpha * m1.weight.data + (1-alpha) * m2.weight.data)
                m3.weight.data = w
                if m1.bias is not None:
                    b = copy.deepcopy(alpha * m1.bias.data + (1-alpha) * m2.bias.data)
                    m3.bias.data = b

        for _ in range(1): # for calculating batchnorm statistics
            train(trainloader, interpolated_model)

        test_loss, test_acc = test(testloader, interpolated_model, criterion)

        accs.append(test_acc)
        alphas.append(alpha)
        print('accs of interpolated models:', accs)

    errors = 100-np.array(accs)

    plt.ylim(-5,70)
    plt.plot(alphas, errors - (errors[0]+errors[-1])/2)

if __name__ == '__main__':
    main()