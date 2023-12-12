import torch

# Load the state dictionary from the .pth file
state_dict = torch.load('unet_state_dict.pth')

# Save the state dictionary to a .pkl file
torch.save(state_dict, 'unet_state_dict.pkl')