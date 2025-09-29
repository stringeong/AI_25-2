import torch
import torchaudio

print("Torch version:", torch.__version__)
print("Torchaudio version:", torchaudio.__version__)
x = torch.rand(3, 3)
print("Tensor:", x)

# MPS(GPU) 사용 가능 여부
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

# 가능하면 MPS로 간단 연산
device = "mps" if torch.backends.mps.is_available() else "cpu"
a = torch.randn(1000, 1000, device=device)
b = a @ a.T
print(device, b.shape, b.dtype)

print(torchaudio.get_audio_backend())
