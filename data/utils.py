import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms.autoaugment import AutoAugment as autoaug
from torchvision.transforms.autoaugment import AutoAugmentPolicy as autoaugpolicy

def get_loader(args):
    print('==> Preparing dataset %s' % args.dataset)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataloader = datasets.CIFAR100
        num_classes = 100         
        batch_size = 128

        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'tiny_imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64,(0.5,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        num_classes = 200
        batch_size = 128

        trainset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        testset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform_test)
        testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)
    
    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if args.arch == 'mlpmixer':
            batch_size = 512
        else:
            batch_size = 256
            
        num_classes = 1000

        trainset = datasets.ImageFolder(f'{args.imagenet_root}/ILSVRC/Data/CLS-LOC/train', transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

        testset = datasets.ImageFolder(f'{args.imagenet_root}/ILSVRC/Data/CLS-LOC/val', transform_test)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    
    return trainloader, testloader