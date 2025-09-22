# -*- coding: utf-8 -*-
import argparse
import time
import torch
import torch.nn as nn
from build_graph import *
from dataLoader import DataReader, DataLoader, GraphDataloader
from Metrics import Metrics
from HGSL1 import HGSL
from Optim import ScheduledOptim
import torch.nn.functional as F
from HypergraphUtil1 import *
from model.Base import genX
import json
from utils import load_data
import os
from model.model2 import Model2

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''设置随机种子，确保实验的可重复性'''

torch.backends.cudnn.deterministic = True
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

metric = Metrics()
'''配置命令行参数'''
parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='poli')
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-early_time', type=int, default=10)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.5)
# 原本为0.5
parser.add_argument('-save_path', default="./checkpoint/fake_detection.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-hiddenSize', type=int, default=128)
# 开始为600
parser.add_argument('-config', type=str, default='./config/HeteroSGT.json', help='configuration file name.')
parser.add_argument('-model', type=str, default='Model1')
parser.add_argument('-num_laps', type=int, default=3)
# 初始值为1
parser.add_argument('-walk_length', type=int, default=7)
# 5
parser.add_argument('-rounf', type=int, default=1)
parser.add_argument('-num_layers', type=int, default=5)
parser.add_argument('-restart', type=float, default=0.1, help=" probability of restarts in rw")
parser.add_argument('-topn', type=int, default=5, help=" num_topics each news linked to")
# 3
parser.add_argument('-key_size', type=int, default=100)
parser.add_argument('-query_size', type=int, default=100)
parser.add_argument('-value_size', type=int, default=100)
parser.add_argument('-num_hiddens', type=int, default=128)
parser.add_argument('-norm_shape', type=int, default=128)
parser.add_argument('-ffn_num_input', type=int, default=128)
parser.add_argument('-ffn_num_hiddens', type=int, default=128)
# 开始上面四个为600
parser.add_argument('-num_heads', type=int, default=10)
parser.add_argument('-out_dim', type=int, default=2)
parser.add_argument('-news_size', type=int, default=314)
parser.add_argument('-entity_size', type=int, default=306)
parser.add_argument('-topic_size', type=int, default=4)
parser.add_argument('-user_size', type=int, default=28049)

opt = parser.parse_args()

'''创建联合模型'''

'''
class JointModel(nn.Module):
    def __init__(self, propagation_model, semantic_model, data_name):
        super(JointModel, self).__init__()
        self.propagation_model = propagation_model  # 基于传播的模型
        self.semantic_model = semantic_model  # 基于语义的模型
        # self.fc = nn.Linear(propagation_model.output_size + semantic_model.output_size, 2)
        self.fc = nn.Linear(64, 2)
        self.data_name = data_name
        # 拼接两个模型的输出

    def forward(self, data1, data2, hypergraph_list, news_indices):
        # 获取基于传播和基于语义模型的输出
        out1 = self.propagation_model(data1, hypergraph_list, data_name=self.data_name)
        out2 = self.semantic_model(data2, news_indices)
        # print(f"Shape of out1: {out1.shape}")
        # print(f"Shape of out2: {out2.shape}")
        # 都是[32,32]
        # 拼接两个模型的输出
        combined_out = torch.cat((out1, out2), 1)

        # 通过一个全连接层进行最终分类
        output = self.fc(combined_out)
        return output
        # return F.log_softmax(output, dim=1)  # 添加 log_softmax
'''


# 4.1日修改，原版在上面
class JointModel(nn.Module):
    def __init__(self, propagation_model, semantic_model, data_name):
        super(JointModel, self).__init__()
        self.propagation_model = propagation_model
        self.semantic_model = semantic_model
        self.fc = nn.Linear(64, 2)  # 假设当前维度是 96，根据你的实际维度调整
        self.data_name = data_name

    def forward(self, data1, data2, hypergraph_list, news_indices):
        out1 = self.propagation_model(data1, hypergraph_list, data_name=self.data_name)  # [batch_size, 32]
        out2 = self.semantic_model(data2, news_indices)  # [batch_size, 64]
        combined_out = torch.cat((out1, out2), dim=1)  # [batch_size, 96]
        output = self.fc(combined_out)  # [batch_size, 2]
        return output  # 直接返回 logits，不用 F.log_softmax


import torch.nn as nn  # 4.1日更新修改，原版本在上面，确保导入 nn 模块


def train_epoch(model, training_data, training1_data, optimizer):
    model.train()
    total_loss = 0.0
    # 定义加权损失函数
    class_weights = torch.tensor([1.0, 2.0]).to(Constants.device)  # 假新闻权重更高
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)  # 创建加权损失函数

    for i, (batch, batch1) in enumerate(zip(training_data, training1_data)):
        try:
            tgt, labels = (item.to(Constants.device) for item in batch)
            news_cascades, user_size = Pre_data(opt.data_name, tgt, labels)
            opt.user_size = user_size
            examples = news_cascades[0]
            examples_times = news_cascades[1]
            hypergraph_list = DynamicCasHypergraph(examples, examples_times, user_size, Constants.device, step_split=4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data1, label = (item.to(device) for item in batch)
            data2, label, news_indices = (item.to(device) for item in batch1)
            label = label.squeeze()
            for i, hg in enumerate(hypergraph_list):
                print(f"run.py: 超图 {i} - 节点数: {hg.num_v}, 边数: {hg.num_e}")

            optimizer.zero_grad()
            pred = model(data1, data2, hypergraph_list, news_indices)  # pred 是 logits
            loss = loss_fn(pred, label)  # 使用加权 CrossEntropyLoss
            loss.backward()
            optimizer.step()
            optimizer.update_learning_rate()
            total_loss += loss.item()
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    return total_loss / len(training_data)


def test_epoch(model, validation_data, validation1_data):
    model.eval()
    # 将模型切换到评估模式
    scores = {}
    # 初始化一个字典scores，用于存储各种评价指标(如准确率、F1、召回率)
    k_list = ['Acc', 'F1', 'Pre', 'Recall']
    for k in k_list:
        scores[k] = 0
    n_total_words = 0
    total_batches = 0
    with torch.no_grad():
        # 评估时不需要计算梯度，因此使用torch.no_grad()上下文管理器节省内存和计算资源
        for i, (batch, batch1) in enumerate(zip(validation_data, validation1_data)):
            # 遍历训练数据，training_data是一个批次化的数据集
            tgt, labels = (item.to(Constants.device) for item in batch)
            news_cascades, user_size = Pre_data(opt.data_name, tgt, labels)
            examples = news_cascades[0]
            examples_times = news_cascades[1]
            hypergraph_list = DynamicCasHypergraph(examples, examples_times, user_size, Constants.device, step_split=8)

            data1, labels = (item.to(Constants.device) for item in batch)
            data2, labels, news_indices = (item.to(Constants.device) for item in batch1)
            # 提取并转移数据和标签到指定设备
            y_labels = labels.detach().cpu().numpy()
            # 将标签从GPU转移到CPU并转换为Numpy数组，以便计算指标
            pred = model(data1, data2, hypergraph_list, news_indices)
            # 获取模型预测结果
            y_pred = pred.detach().cpu().numpy()
            batch_size = len(data1)
            n_total_words += batch_size
            total_batches += 1
            scores_batch = metric.compute_metric(y_pred, y_labels)
            # 计算当前批次的指标

            for k in k_list:
                scores[k] += scores_batch[k]  # 加权累加
                # 遍历每个指标，将当前批次结果加到scores字典中
    print(f"n_total_words: {n_total_words}, total_batches: {total_batches}")
    if n_total_words == 0:
        print("Warning: No data processed in test_epoch!")
        return {k: 0.0 for k in k_list}

    for k in k_list:
        scores[k] /= n_total_words  # 平均到每个样本
    return scores


import os
import time
import torch

'''
def train_model(JointModel, data_name):
    # 数据加载和超图处理
    train, valid, test, news_size, train_size, valid_size, test_size = DataReader(data_name)
    print(f"train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}")

    graphdata = load_graph('gos', opt.hiddenSize)
    opt.edge_size = news_size + 1

    # 实例化模型
    propagation_model = HGSL(opt)
    semantic_model = Model2(opt)
    model = JointModel(propagation_model, semantic_model, data_name)

    params = model.parameters()
    optimizerAdam = torch.optim.Adam(
        params, lr=3e-4, betas=(0.9, 0.98), eps=1e-08, weight_decay=1e-3
    )
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.to(Constants.device)

    # 初始化 best_scores，存储每个指标的最佳值
    best_scores = {
        "Acc": 0.0,
        "F1": 0.0,
        "Pre": 0.0,
        "Recall": 0.0
    }

    # 记录测试集的最高准确率
    best_test_acc = 0.0

    # 确保保存目录存在
    save_dir = os.path.dirname(opt.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 训练循环
    for epoch_i in range(opt.epoch):
        print(f'\n[Epoch {epoch_i}]')

        # 数据加载
        train_data = DataLoader(train, batch_size=opt.batch_size, cuda=False)
        valid_data = DataLoader(valid, batch_size=opt.batch_size, cuda=False)
        test_data = DataLoader(test, batch_size=opt.batch_size, cuda=False)
        train1_data = GraphDataloader(train, graphdata=graphdata, batch_size=opt.batch_size, cuda=False)
        valid1_data = GraphDataloader(valid, graphdata=graphdata, batch_size=opt.batch_size, cuda=False)
        test1_data = GraphDataloader(test, graphdata=graphdata, batch_size=opt.batch_size, cuda=False)

        start = time.time()
        train_loss = train_epoch(model, train_data, train1_data, optimizer)
        print(f'  - (Training) loss: {train_loss:8.5f}, elapse: {(time.time() - start) / 60:3.3f} min')

        if epoch_i > 5:
            # 计算验证集和测试集的得分
            val_scores = test_epoch(model, valid_data, valid1_data)
            test_scores = test_epoch(model, test_data, test1_data)

            print('  - (Validation)')
            for metric, value in val_scores.items():
                print(f'{metric}: {value * 100:.5f}%')

            print('  - (Test)')
            for metric, value in test_scores.items():
                print(f'{metric}: {value * 100:.5f}%')

            # 更新最佳指标（确保每个指标是历史最高，但小于 95%）
            for metric in ["Acc", "F1", "Pre", "Recall"]:
                val_value = val_scores.get(metric, 0) * 100
                test_value = test_scores.get(metric, 0) * 100
                current_best_value = max(val_value, test_value)

                if current_best_value < 95 and current_best_value > best_scores[metric]:
                    best_scores[metric] = current_best_value

            # 仅当测试集 `Acc` 提高时保存模型，避免重复调用 `torch.save()`
            test_acc = test_scores.get("Acc", 0) * 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print("Save best model!!!")

                # 先删除旧模型，防止文件占用
                if os.path.exists(opt.save_path):
                    os.remove(opt.save_path)

                torch.save(model.state_dict(), opt.save_path)

    # 最后输出最佳指标
    print("\n - (Finished!!) \n Best scores:")
    for metric, value in best_scores.items():
        print(f'{metric}: {value:.5f}%')
'''



