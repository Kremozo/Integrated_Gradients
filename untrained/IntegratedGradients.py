import torch
import torch.nn as nn
from typing import List

def integrated_gradients(inputs: torch.Tensor, model: nn.Module, target_class, baseline=None,steps = 50):
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    #interpolate steps
    scaled_inputs: List[torch.Tensor] = [baseline+ (float(i) / steps) * (inputs-baseline) for i in range(steps+1)]
    grads = []
    
    #calculate grads
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad_()
        model.zero_grad()
        output = model(scaled_input)
        target: torch.Tensor = output[0,target_class]
        grad = torch.autograd.grad(
            outputs=target,
            inputs=scaled_input,
            retain_graph=False,
            create_graph=False
        )[0]
        grads.append(grad)
    
    #Riemann sum
    grads_tensor = torch.stack(grads)
    avg_grads = grads_tensor.mean(dim=0)

    attributions = (inputs - baseline) * avg_grads
    return attributions

