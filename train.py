import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np

from models.utils import get_model
from data.utils import get_loader
from optim.utils import get_optim
from utils import Logger, AverageMeter, prune, mask_transfer, sign_transfer
from tqdm import tqdm

import argparse
import os
import random
import copy
import datetime

parser = argparse.ArgumentParser()

# Datasets`
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--imagenet_root', default='./data', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--warm_up', type=int, default=10)

parser.add_argument('--name', help='name of the experiment')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn')
# Miscs
parser.add_argument('--seed', type=int, help='manual seed', default=1)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--pruning_iters', type=int, default=0)
parser.add_argument('--pretrained_dir', type=str, default=None)
parser.add_argument('--aws', action='store_true', default=False)
parser.add_argument('--sign_transfer', action='store_true', default=False)
parser.add_argument('--mask_transfer', action='store_true', default=False)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def main():
    if args.name is None:
        args.name = datetime.datetime.strptime('2020-01-07 15:40:15', '%Y-%m-%d %H:%M:%S') + f'_{args.dataset}_{args.arch}'

    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)

    # Model
    # Declare the model with a random seed to ensure different initializations
    model = get_model(args)
    model = model.cuda()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    trainloader, testloader = get_loader(args)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()

    initial_params = copy.deepcopy(model.state_dict())

    logger = Logger(os.path.join(f'checkpoint/{args.name}', 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    is_pruning_stage = (args.pruning_iters > 1)

    # Training a sparse network from scratch
    if not is_pruning_stage:
        assert args.pruning_iters == 0

        pretrained = torch.load(args.pretrained_dir)['state_dict']
        model.load_state_dict(pretrained)

        if args.sign_transfer:
            # transfer signed mask
            sign_transfer(copy.deepcopy(model.state_dict()), initial_params, model)
        elif args.mask_transfer:
            # transfer binary mask
            mask_transfer(copy.deepcopy(model.state_dict()), initial_params, model)

        total = 0
        pruned = 0
        for m in model.modules():
            if hasattr(m, 'fixed_m'):
                total += m.weight.numel()
                pruned += (m.fixed_m==0).float().sum()
        pr = pruned / total

        print('Pruning Ratio:', round(pr.item()*100,2))

    # Train and val
    for iteration in range(args.pruning_iters+1):
        print(f'PRUNING ITERATION {iteration}/{args.pruning_iters}')

        if iteration == 0 and is_pruning_stage:
            epochs = args.warm_up
        else:
            epochs = args.epochs

        optimizer, scheduler = get_optim(args,model)

        for epoch in range(epochs):
            print(args.name)
            print('\nEpoch: [%d | %d]' % (epoch + 1, epochs))

            train_loss, train_acc = train(trainloader, model, criterion, optimizer, args)
            test_loss, test_acc = test(testloader, model, criterion)

            if not is_pruning_stage:
                scheduler.step()

            logger.append([scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc])

            print(f'Train Acc: {train_acc}\nTest Acc: {test_acc}')

        if is_pruning_stage:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=f'checkpoint/{args.name}', filename=f'checkpoint_iter{iteration}.pth.tar')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=f'checkpoint/{args.name}', filename=f'checkpoint.pth.tar')
        
        if is_pruning_stage and iteration < args.pruning_iters:
            pr = prune(model, iteration, args.pruning_iters)
        else:
            total = 0
            pruned = 0
            for m in model.modules():
                if hasattr(m, 'fixed_m'):
                    total += m.weight.numel()
                    pruned += (m.fixed_m==0).float().sum()
            pr = pruned / total
        
        print('Pruning Ratio:', round(pr.item()*100,2))

    logger.close()
    logger.plot()

def train(trainloader, model, criterion, optimizer, args):
    model.train()

    losses = AverageMeter()
    acc = AverageMeter()

    pbar = tqdm(trainloader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        if args.aws:
            alpha = random.random()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.alpha = alpha
        else:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.alpha = 0

        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        correct = (outputs.max(1)[1]).eq(targets).sum()
        losses.update(loss.item(), inputs.size(0))
        acc.update(correct.item()/inputs.size(0)*100, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=losses.avg, acc=acc.avg)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.alpha = 0

    return (losses.avg, acc.avg)

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

def save_checkpoint(state, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    main()
