import torch
import time

# 큰 텐서 만들기 (20000 x 20000)
size = 20000
a = torch.rand(size, size)
b = torch.rand(size, size)

# CPU 연산
device = torch.device("cpu")
a_cpu = a.to(device)
b_cpu = b.to(device)

start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)  # 행렬 곱
torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()
print(f"CPU time: {end - start:.4f} seconds")

# MPS 연산 (맥북 GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    a_mps = a.to(device)
    b_mps = b.to(device)

    start = time.time()
    c_mps = torch.matmul(a_mps, b_mps)
    torch.mps.synchronize()  # MPS 연산 동기화
    end = time.time()
    print(f"MPS time: {end - start:.4f} seconds")
else:
    print("MPS backend is not available on this system.")