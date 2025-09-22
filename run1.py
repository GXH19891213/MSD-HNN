# -*- coding: utf-8 -*-

import argparse
import time
import torch
from dataLoader import DataReader,  DataLoader
from Metrics import Metrics
from HGSL1 import HGSL
from Optim import ScheduledOptim
import torch.nn.functional as F
from HypergraphUtil1 import *
from model.model1 import Model1, train
import json
from utils import load_data
import os

# 设置随机种子，确保实验的可重复性
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()

# 配置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='poli')
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-early_time', type=int, default=10)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-save_path', default="./checkpoint/fake_detection.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('hiddenSize', type=int, default=600)
parser.add_argument('--config', type=str, default='./config/HeteroSGT.json', help='configuration file name.')
parser.add_argument('-model', type=str, default='Model1')
parser.add_argument('-num_laps', type=int, default=1)
parser.add_argument('-walk_length', type=int, default=5)
parser.add_argument('-rounf', type=int, default=1)
parser.add_argument('-num_layers', type=int, default=5)
parser.add_argument('-restart', type=float, default=0.1, help=" probability of restarts in rw")
parser.add_argument('-topn', type=int, default=3, help=" num_topics each news linked to")
opt = parser.parse_args()

def train_epoch(model, training_data, hypergraph_list, optimizer):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(training_data):
        tgt, labels = (item.to(Constants.device) for item in batch)
        '''
        for tgt, labels in zip(tgt, labels):
        news_cascades = Pre_data(data_name, tgt, labels)
        examples = news_cascades[0]
        examples_times = news_cascades[1]
        hypergraph_list = DynamicCasHypergraph(examples, examples_times, user_size, Constants.device, step_split=8)
        '''
        optimizer.zero_grad()

        # 执行模型的前向传播
        pred = model(tgt, hypergraph_list)
        loss = F.nll_loss(pred, labels.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        total_loss += loss.item()

    return total_loss

def train_model(HGSL, data_name):
    # 数据加载和超图处理
    train, valid, test, news_size = DataReader(data_name)
    user_size = len(train)  # 假设用户大小等于训练集的长度
    # 使用DynamicCasHypergraph来生成hypergraph_list
    news_cascades=Pre_data(data_name, news_size)
    examples=news_cascades[0]
    examples_times=news_cascades[1]
    # hypergraph_list = DynamicCasHypergraph(data_name, user_size, Constants.device, step_split=8)
    hypergraph_list = DynamicCasHypergraph(examples, examples_times, Constants.device, step_split=8)

    # 使用DataLoader加载数据，DataLoader用于将数据集分批次地提供给模型进行训练或者测试
    train_data = DataLoader(train, batch_size=opt.batch_size, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, cuda=False)

    opt.user_size = user_size
    opt.edge_size = news_size + 1

    # 实例化模型
    model = HGSL(opt)
    params = model.parameters()

    # 使用Adam优化器
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    # 如果有GPU，移动模型到GPU
    if torch.cuda.is_available():
        model = model.to(Constants.device)

    validation_history = 0.0
    best_scores = {}

    # 训练周期
    for epoch_i in range(opt.epoch):
        print(f'\n[Epoch {epoch_i}]')
        start = time.time()
        train_loss = train_epoch(model, train_data, hypergraph_list, optimizer)
        print(f'  - (Training) loss: {train_loss:8.5f}, elapse: {(time.time() - start) / 60:3.3f} min')

        if epoch_i > 5:
            # 计算验证集和测试集的得分
            scores = test_epoch(model, valid_data, hypergraph_list)
            print('  - (Validation)')
            for metric in scores.keys():
                print(f'{metric}: {scores[metric] * 100:.5f}%')

            print('  - (Test)')
            scores = test_epoch(model, test_data, hypergraph_list)
            for metric in scores.keys():
                print(f'{metric}: {scores[metric] * 100:.5f}%')

            # 保存最佳模型
            if validation_history <= sum(scores.values()):
                print(f"Best Test Accuracy: {scores['Acc'] * 100:.5f}% at Epoch: {epoch_i}")
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)

    print(" - (Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(f'{metric}: {best_scores[metric] * 100:.5f}%')

def test_epoch(model, validation_data, hypergraph_list):
    # test_epoch负责在验证集上validation_data上进行模型的评估
    # model：训练好的模型
    # validation_data：验证集数据，通常是批次化的验证数据
    # hypergraph_list:包含了超图结构的列表
    model.eval()
    # 设置为评估模式
    scores = {}
    k_list = ['Acc', 'F1', 'Pre', 'Recall']
    for k in k_list:
        scores[k] = 0

    n_total_words = 0
    # 用于统计验证集中的样本总数，用于对评估指标进行加权平均
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            # 遍历validation_data中每一个批次
            tgt, labels = (item.to(Constants.device) for item in batch)
            y_labels = labels.detach().cpu().numpy()
            pred = model(tgt, hypergraph_list)
            y_pred = pred.detach().cpu().numpy()

            n_total_words += len(tgt)
            scores_batch = metric.compute_metric(y_pred, y_labels)

            for k in k_list:
                scores[k] += scores_batch[k]

    for k in k_list:
        scores[k] = scores[k] / n_total_words
    return scores

if __name__ == "__main__":
    # 在主程序中调用train_model并传入HGSL作为模型
    train_model(HGSL, opt.data_name)


