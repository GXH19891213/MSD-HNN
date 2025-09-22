import torch
print(torch.__version__)          # PyTorch 版本（需 ≥1.10.0）
print(torch.version.cuda)          # PyTorch 对应的 CUDA 版本（需为 11.5）
print(torch.cuda.is_available())   # 必须返回 True