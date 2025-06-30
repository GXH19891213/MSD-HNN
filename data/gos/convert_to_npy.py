import numpy as np

# 从 train_idx.txt 读取数据
with open('test_idx.txt', 'r') as f:
    data = [int(line.strip()) for line in f if line.strip()]

# 转换为 NumPy 数组并保存为 .npy 文件
np_array = np.array(data, dtype=np.int32)  # 可根据需要调整 dtype（如 np.int64）
np.save('test_idx.npy', np_array)

print("转换完成！保存为 train_idx.npy")