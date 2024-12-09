import torch

checkpoint = torch.load('runs/gev/best_byol_model.pth', map_location=torch.device('cpu'))

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

for key in state_dict.keys():
    print(key)