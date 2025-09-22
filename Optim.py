'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''
 # 一个简单的包装器类，用于学习率调度
    def __init__(self, optimizer, d_model, n_warmup_steps):
        # optimizer：一个传入的优化器实例：如torch.optim.Adam或torch.optim.SGD
        self.optimizer = optimizer
        self.d_model = d_model
        # d_model:模型的维度，用于计算学习率的调整策略
        self.n_warmup_steps = n_warmup_steps
        # 预热步骤数，在这些步骤中，学习率会从一个较小的值组件增加到预定的学习率
        self.n_current_steps = 0
        # 当前步骤计数，用于跟踪训练过程中的步骤数
    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()
        # 调用内部优化器的step方法，执行梯度更新

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()
        # 调用内部优化器的zero_grad方法，用于在每次迭代开始时清零梯度

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        # 每调用一次这个方法，当前步骤数增加1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
