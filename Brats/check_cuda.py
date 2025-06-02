import torch

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA verfügbar: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'CUDA Geräte: {torch.cuda.device_count()}')
    print(f'CUDA Gerätename: {torch.cuda.get_device_name(0)}')
    
    # Erstelle einen kleinen Tensor und führe eine Operation auf der GPU aus
    x = torch.rand(5, 3).cuda()
    print(f'Tensor auf GPU: {x.device}')
    print(f'Ergebnis einer einfachen Operation: {x.sum().item()}')
else:
    print('CUDA ist nicht verfügbar. Verwende CPU.')
    
    # Prüfe, warum CUDA möglicherweise nicht verfügbar ist
    print(f'PyTorch Build Info: {torch.__config__.show()}') 