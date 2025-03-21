import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models import efficientnet, mobilenetv2, shufflenet, mlp_mixer

def get_model(args):
    if args.dataset == 'cifar100':
        num_classes = 100
        if args.arch == 'resnet50':
            model = tv_models.resnet.resnet50(num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        elif args.arch == 'mobilenetv2':
            model = mobilenetv2.mobilenetv2(num_classes=num_classes)
        elif args.arch == 'efficientnet':
            model = efficientnet.EfficientNetB0(num_classes)
        elif args.arch == 'shufflenet':
            model = shufflenet.shufflenet(num_classes=num_classes)

    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
        if args.arch == 'resnet50':
            model = tv_models.resnet.resnet50(num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        elif args.arch == 'mobilenetv2':
            model = mobilenetv2.mobilenetv2(num_classes=num_classes)
        elif args.arch == 'efficientnet':
            model = efficientnet.EfficientNetB0(num_classes)
        elif args.arch == 'shufflenet':
            model = shufflenet.shufflenet(num_classes=num_classes)
    
    elif args.dataset == 'imagenet':
        num_classes = 1000

        if args.arch == 'resnet50':
            model = tv_models.resnet.resnet50(num_classes=num_classes)
        if args.arch == 'mobilenetv2':
            model = tv_models.mobilenetv2.mobilenet_v2(num_classes=num_classes, dropout=0)
        if args.arch == 'mlpmixer':
            model = mlp_mixer.MLPMixer(image_size=224, channels=3, patch_size=16, dim=512, depth=12, num_classes=1000)

    for m in model.modules():
        if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.Conv1d):
                pass
            else:
                print(f"{type(m)} is not supported")
                import pdb; pdb.set_trace()
        if isinstance(m, nn.Conv2d):
            setattr(m, 'fixed_m', nn.Parameter(torch.ones(m.weight.shape),requires_grad=False))
            setattr(m, 'use_bias', m.bias is not None)
            new_method = new_conv_forward.__get__(m, m.__class__)
            setattr(m, 'forward', new_method)
        if isinstance(m, nn.Linear) and not hasattr(m, 'no_prune'):
            if m.weight.shape[0] != num_classes:
                setattr(m, 'fixed_m', nn.Parameter(torch.ones(m.weight.shape),requires_grad=False))
                setattr(m, 'use_bias', m.bias is not None)
                new_method = new_linear_forward.__get__(m, m.__class__)
                setattr(m, 'forward', new_method)
        if isinstance(m, nn.Conv1d) and not hasattr(m, 'no_prune'):
            if m.weight.shape[0] != num_classes:
                setattr(m, 'fixed_m', nn.Parameter(torch.ones(m.weight.shape),requires_grad=False))
                setattr(m, 'use_bias', m.bias is not None)
                new_method = new_conv1d_forward.__get__(m, m.__class__)
                setattr(m, 'forward', new_method)
        if isinstance(m, nn.BatchNorm2d):
            setattr(m, 'alpha', 0)
            new_method = new_bn_forward.__get__(m, m.__class__)
            setattr(m, 'forward', new_method)
        if isinstance(m, nn.LayerNorm):
            setattr(m, 'alpha', 0)
            new_method = new_ln_forward.__get__(m, m.__class__)
            setattr(m, 'forward', new_method)
    
    return model


def new_conv_forward(self, x):
    if self.use_bias:
        return F.conv2d(x, self.weight * self.fixed_m.detach(), self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
    else:
        return F.conv2d(x, self.weight * self.fixed_m.detach(), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

def new_conv1d_forward(self, x):
    if self.use_bias:
        return F.conv1d(x, self.weight * self.fixed_m.detach(), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
    else:
        return F.conv1d(x, self.weight * self.fixed_m.detach(), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

def new_linear_forward(self, x):
    if self.use_bias:
        return F.linear(x, self.weight * self.fixed_m.detach(), self.bias)
    else:
        return F.linear(x, self.weight * self.fixed_m.detach())

def new_bn_forward(self, x):
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        assert self.num_batches_tracked is not None
        self.num_batches_tracked.add_(1)
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
    
    if self.training:
        bn_training = True
    else:
        bn_training = (self.running_mean is None) and (self.running_var is None)
    
    # Scale and shift
    gamma = (self.alpha + (1-self.alpha) * self.weight)
    beta = ((1-self.alpha) * self.bias)
    out = F.batch_norm(x, self.running_mean, self.running_var, gamma, beta, bn_training, exponential_average_factor, self.eps)
    
    return out
def new_ln_forward(self, x):
    gamma = (self.alpha + (1-self.alpha) * self.weight)
    beta = ((1-self.alpha) * self.bias)
    out = F.layer_norm(x, self.weight.shape, gamma, beta)
    return out