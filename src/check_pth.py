import torch

state_dict = torch.load('data/pretrain_res50x1.pth', map_location=torch.device('cpu'))
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
weights_exist = any('weight' in key for key in state_dict.keys())
if weights_exist:
    print("The .pth file contains model weights.")
else:
    print("No model weights found in the .pth file.")
