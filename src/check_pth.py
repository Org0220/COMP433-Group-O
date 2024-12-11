import torch

state_dict = torch.load('runs/dave/best_byol_model.pth', map_location=torch.device('cpu'))

print("Keys in state_dict:")
for key in state_dict.keys():
    if 'model_state_dict' in key:
        model_state_dict = state_dict[key]
        break

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
weights_exist = any('weight' in key for key in state_dict.keys())
if weights_exist:
    print("The .pth file contains model weights.")
else:
    print("No model weights found in the .pth file.")
