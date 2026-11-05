import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    # Perform a simple GPU calculation to verify functionality (Smoke Test)
    x = torch.randn(256, 256, device='cuda')
    result = x @ x.T
    torch.cuda.synchronize()
    print("GPU calculation successful, result shape:", result.shape)