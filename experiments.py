#%%

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model 
import argparse
import pickle
import torch
import numpy as np
# %%



import os
if os.path.exists('args.pkl'):
    with open('args.pkl', 'rb') as f:
        args = pickle.load(f)


# %%
args.in_channels = 3
args.num_classes = 10
model, args = load_model(args)

# %%

model_list = [model, model, model, model, model]
# %%

weight_dict = {}
for name, param in model.named_parameters():
    if name not in weight_dict:
        weight_dict[name] = []
    weight_dict[name].append(param.data)

# %%

for name in weight_dict:
    weights = torch.stack(weight_dict[name])
    mean = torch.mean(weights, dim=0)
    std = torch.std(weights, dim=0)
    print("mean shape: ", mean.shape)
    print("std shape: ", std.shape)
    
    # sample from this distribution same shape
    sample = torch.normal(mean, std)
    print("sample shape: ", sample.shape)
    break
    
# %%
