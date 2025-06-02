import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Anzahl CUDA-Geräte: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA-Gerät {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA ist nicht verfügbar.") 