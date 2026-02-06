import torch

device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")