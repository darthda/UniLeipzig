import torch
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")