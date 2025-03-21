import torch
import torch.nn as nn
import copy

def sign_transfer(pretrained, init_sd, model, winit_only=False):
        model.load_state_dict(pretrained)

        signs = []
        b_signs = []

        bn_w = []
        bn_b = []

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and winit_only:
                bn_w.append(m.weight.data.clone())
                bn_b.append(m.bias.data.clone())

            if hasattr(m, 'fixed_m'):
                sign = torch.sign(m.weight.data*m.fixed_m.data).clone()
                if m.use_bias:
                    b_sign = torch.sign(m.bias.data).clone()
                    b_signs.append(b_sign)
                signs.append(sign)
            elif isinstance(m, nn.Linear):
                fc_w_sign = torch.sign(m.weight.data).clone()
                if m.bias is not None:
                    fc_b_sign = torch.sign(m.bias.data).clone()

        model.load_state_dict(init_sd)

        i=0
        j=0

        bn_i=0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and winit_only:
                m.weight.data = bn_w[bn_i]
                m.bias.data = bn_b[bn_i]
                bn_i += 1
            if hasattr(m, 'fixed_m'):
                m.weight.data = abs(m.weight.data) * signs[i]
                m.fixed_m.data = (signs[i].cuda() != 0)
                if m.use_bias:
                    m.bias.data = abs(m.bias.data) * b_signs[j]
                    j+=1
                i+=1
            elif isinstance(m, nn.Linear):
                m.weight.data = abs(m.weight.data) * fc_w_sign
                if m.bias is not None:
                    m.bias.data = abs(m.bias.data) * fc_b_sign

def mask_transfer(pretrained, init_sd, model):
    model.load_state_dict(pretrained)

    masks = []
    for m in model.modules():
        if hasattr(m, 'fixed_m'):
            mask = (m.fixed_m.data==1).clone()
            masks.append(mask)

    model.load_state_dict(init_sd)

    i=0
    for m in model.modules():
        if hasattr(m, 'fixed_m'):
            m.fixed_m.data = masks[i]
            i+=1