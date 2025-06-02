import torch
import gc

# Speicherinformationen vor der Bereinigung
print(f"Vor der Bereinigung:")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# GPU-Cache leeren
torch.cuda.empty_cache()

# Garbage Collection forcieren
gc.collect()

# CUDA-Speicher zurücksetzen
torch.cuda.reset_peak_memory_stats()
if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
    torch.cuda.reset_accumulated_memory_stats()
if hasattr(torch.cuda, 'reset_max_memory_allocated'):
    torch.cuda.reset_max_memory_allocated()

# Speicherinformationen nach der Bereinigung
print(f"\nNach der Bereinigung:")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Überprüfen, ob CUDA verfügbar ist und welche GPU verwendet wird
if torch.cuda.is_available():
    print(f"\nCUDA ist verfügbar. Verfügbare GPUs: {torch.cuda.device_count()}")
    print(f"Aktuelle GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU-Eigenschaften: {torch.cuda.get_device_properties(0)}")
else:
    print("\nCUDA ist nicht verfügbar.")

print("\nGPU-Speicher sollte jetzt bereinigt sein.") 