try:
    import torch
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Anzahl CUDA-Geräte: {torch.cuda.device_count()}")
        print(f"Aktuelles CUDA-Gerät: {torch.cuda.current_device()}")
        print(f"CUDA-Gerätename: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA ist nicht verfügbar.")
except ImportError:
    print("PyTorch ist nicht installiert. Installiere es mit 'pip install torch'.") 