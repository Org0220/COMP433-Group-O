# convert_checkpoint.py

import pickle
import torch
import numpy as np
import re
from torchvision.models import resnet50
from haiku._src.data_structures import frozendict  

def load_jax_checkpoint(filename):

    print("Loading JAX checkpoint...")
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print("Checkpoint loaded.")
    print("Top-level keys in JAX parameters:")
    print(params.keys())
    return params

def extract_params(params_dict, parent_key='', result_dict=None):

    if result_dict is None:
        result_dict = {}
    for key, value in params_dict.items():
        full_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, (dict, frozendict)):
            extract_params(value, full_key, result_dict)
        else:
            result_dict[full_key] = value
    return result_dict

def map_jax_key_to_pytorch_key(jax_key):
    
    
    key = jax_key.replace('~/', '')
    
    key = key.lstrip('/')
    parts = key.split('/')

    pytorch_key = ''
    param_name = ''

    if parts[0] == 'res_net50':
        parts = parts[1:]

    if parts[0] == 'initial_conv':
        pytorch_key = 'conv1.weight'
    elif parts[0] == 'initial_batchnorm':
        if parts[1] == 'scale':
            pytorch_key = 'bn1.weight'
        elif parts[1] == 'offset':
            pytorch_key = 'bn1.bias'
    else:
        layer_match = re.match(r'block_group_(\d+)', parts[0])
        if layer_match:
            layer_idx = int(layer_match.group(1)) + 1  
            block_match = re.match(r'block_(\d+)', parts[1])
            if block_match:
                block_idx = int(block_match.group(1))
                sublayer = parts[2]
                if sublayer.startswith('conv_'):
                    conv_idx = int(sublayer[len('conv_'):]) + 1  
                    pytorch_key = f'layer{layer_idx}.{block_idx}.conv{conv_idx}.weight'
                elif sublayer.startswith('batchnorm_'):
                    bn_idx = int(sublayer[len('batchnorm_'):]) + 1  
                    param_type = parts[3]
                    if param_type == 'scale':
                        pytorch_key = f'layer{layer_idx}.{block_idx}.bn{bn_idx}.weight'
                    elif param_type == 'offset':
                        pytorch_key = f'layer{layer_idx}.{block_idx}.bn{bn_idx}.bias'
                elif sublayer == 'shortcut_conv':
                    pytorch_key = f'layer{layer_idx}.{block_idx}.downsample.0.weight'
                elif sublayer == 'shortcut_batchnorm':
                    param_type = parts[3]
                    if param_type == 'scale':
                        pytorch_key = f'layer{layer_idx}.{block_idx}.downsample.1.weight'
                    elif param_type == 'offset':
                        pytorch_key = f'layer{layer_idx}.{block_idx}.downsample.1.bias'
    return pytorch_key

def convert_jax_to_pytorch(params):

    print("\nConverting parameters from JAX to PyTorch format...")
    
    flat_params = extract_params(params)
    
    mapped_params = {}
    missing_keys = []
    unexpected_keys = []
    
    for jax_key, value in flat_params.items():
        pytorch_key = map_jax_key_to_pytorch_key(jax_key)
        if pytorch_key == '':
            print(f"Could not map JAX key: {jax_key}")
            continue
        tensor = torch.tensor(np.array(value))
        
        if 'conv' in pytorch_key and len(tensor.shape) == 4:
            tensor = tensor.permute(3, 2, 0, 1)  
        elif 'downsample.0.weight' in pytorch_key and len(tensor.shape) == 4:
            tensor = tensor.permute(3, 2, 0, 1)
        elif 'bn' in pytorch_key or 'downsample.1' in pytorch_key:
            tensor = tensor.view(-1)
        elif 'fc.weight' in pytorch_key and len(tensor.shape) == 2:
            tensor = tensor.t()
        
        mapped_params[pytorch_key] = tensor

    model = resnet50()
    model_state_dict = model.state_dict()
    
    new_state_dict = {}
    for key in model_state_dict.keys():
        if key in mapped_params:
            new_state_dict[key] = mapped_params[key]
        else:
            new_state_dict[key] = model_state_dict[key]
            if key not in ['fc.weight', 'fc.bias']:
                missing_keys.append(key)
    
    print(f"\nConversion completed with {len(missing_keys)} missing keys.")
    if missing_keys:
        print("Missing keys:")
        for key in missing_keys:
            print(key)

    for key in mapped_params.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    if unexpected_keys:
        print("\nUnexpected keys in mapped_params:")
        for key in unexpected_keys:
            print(key)
    
    return new_state_dict

def save_pytorch_model(state_dict, filename):
    print(f"\nSaving PyTorch model to {filename}...")
    torch.save(state_dict, filename)
    print("Model saved.")

def main():

    checkpoint_filename = 'data/pretrain_res50x1.pkl'
    params = load_jax_checkpoint(checkpoint_filename)
    
    if isinstance(params['experiment_state'], list) or isinstance(params['experiment_state'], tuple):
        model_params = params['experiment_state'][0]
    else:
        model_params = params['experiment_state']

    print("Type of model_params:", type(model_params))
    print("Keys in model_params:")
    print(model_params.keys())
    
    pytorch_state_dict = convert_jax_to_pytorch(model_params)
    
    model = resnet50()
    model.load_state_dict(pytorch_state_dict, strict=False)
    
    output_filename = 'data/pretrain_res50x1.pth'
    save_pytorch_model(model.state_dict(), output_filename)

if __name__ == "__main__":
    main()
