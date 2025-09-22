import torch
PAD = 0
# 定义常量PAD,值被设为0，用于表示填充
step_split = 2
# 用于控制某个过程的步长或者分割数据的尺寸
n_heads = 14
# 多头自注意力机制中的头数，在多头注意力中，14个头可以并行处理不同的表示子空间
#cate = ['retweet', 'support', 'deny']
cate = ['retweet']
early_type = 'time' # 'engage' or 'time'
# 定义一个用于指定基于时间或者用户互动的某种早期处理或检测的类型
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
