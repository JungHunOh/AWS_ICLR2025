import torch
import numpy as np

def prune(model, iteration, total_iters):
    total = 0
    pruned = 0
    pruning_ratio = 0.2
    weights = []
    for m in model.modules():
        if hasattr(m, 'fixed_m'):
            mask = m.fixed_m.reshape(-1)
            weights.append(m.weight.reshape(-1)[mask==1])
    
    weights = torch.cat(weights,dim=0)
    topk = abs(weights).topk(int(weights.shape[0]*(1-pruning_ratio)))[0][-1].cpu().detach().numpy()

    for m in model.modules():
        if hasattr(m, 'fixed_m'):
            weights = m.weight.data.cpu().reshape(-1).numpy()
            fixed_m = m.fixed_m.cpu().reshape(-1).numpy()

            num_total = weights.shape[0]
            fixed_m[fixed_m!=0] = (abs(weights)>=topk)[fixed_m!=0].astype(np.float32)

            new_mask = fixed_m.reshape(m.weight.shape)
        
            m.fixed_m.data = torch.from_numpy(new_mask).cuda()
            total+=num_total
            pruned+=(m.fixed_m==0).sum()

    print('pruned_ratio:',pruned / total)

    return pruned / total