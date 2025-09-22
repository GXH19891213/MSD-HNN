import torch
import dhg
import pickle
import os
import numpy as np

from Data_preprocessing import *
'''
CascadeHypergraph根据单个时间段的级联数据构建一个超图，
将一个级联数据集合转换为一个超图对象，其中每个超图的边代表一组共同互动的用户
'''
def CascadeHypergraph(cascades, user_size, device):
    # 根据级联数据构建超图，cascades：一个包含级联数据的列表，表示不同用户的互动行为
    # user_size:用户总数 device：设备信息
    # cascades = cascades.tolist()
    edge_list = []
    # 创建空列表edge_list，该列表将存储超图的边，每个边是一个用户的集合，表示在一个级联中的所有用户
    for cascade in cascades:
        cascade = set(cascade)
        # 将级联中用户转换成集合，去除重复用户
        if len(cascade) > 2:
            cascade.discard(0)
            # 如果级联用户数大于2，删除用户ID为0的用户，ID为0是特殊标识符，不是有效用户，因此需要删除
        edge_list.append(cascade)
        # print(f'edge_list: {edge_list}')
        # 将处理后的用户集合添加到边列表edge_list中
        # print(f"user_size:{user_size}, device:{device}")
    cascade_hypergraph = dhg.Hypergraph(user_size, edge_list, device=device)
    # user_size:定义超图节点数量
    # edge_list 定义超图的边(每一个边是一个用户集合)
    return cascade_hypergraph

def DynamicCasHypergraph(examples, examples_times, user_size, device, step_split=4):
    # 根据级联和时间戳数据，动态地将数据划分为多个超图序列
    # param examples: 级联（用户），一个包含用户列表
    # param examples_times: 级联时间戳（用户参与级联的时间）
    # param user_size: 数据集中的所有用户总数
    # param device: 所在设备
    # param step_split: 划分几个超图
    # return: 超图序列


    # print(examples[200])
    hypergraph_list = []
    # 用于存储最终生成多少个超图对象
    time_sorted = []
    # 用于存储所有时间戳,便于后续排序和划分
    for time in examples_times:
        time_sorted += time
        # time_sorted += time[:-1]
        # 遍历每个级联的时间戳列表
        # 从每个时间戳列表中去掉最后一个时间戳，假设最后一个时间戳是结束符
        # 将每个级联的时间戳加入 time_sorted列
    time_sorted = sorted(time_sorted)   # 将所有时间戳升序排列
    # len(time_sorted:2853)
    split_length = len(time_sorted) // step_split    # 每个时间段长度
    # len(split_length) 356
    # step_split是划分超图的数量
    start_time = 0
    end_time = 0
    total_time_points = len(time_sorted)
    # print(f"总时间戳数: {total_time_points}, 分段长度: {split_length}")
    # print(f"排序后时间戳: {time_sorted}")
    for x in range(split_length, split_length * step_split, split_length):
        # range(start,stop,step):生成一个序列，从start开始，到stop之前，步长为step
        # 从第一个时间段长度开始，循环直到总的时间戳数结束，每次循环增加一个时间段的长度
        # 如 range(100, 800, 100)  # 生成 [100, 200, 300, ..., 700]
        # 打印关键参数辅助调试
        start_time = end_time
        # 将上个时间段的结束时间end_time赋值给当前时间段的起始时间start_time,确保当前时间段的起点是上一个时间段的终点
        end_time = time_sorted[x]
        # print(f"start time: {start_time}")
        # print(f"end time: {end_time}")

        # 从已经排序的时间戳列表time_sorted中，取出序列位置x对应的时间戳，赋值为当前时间段的结束时间end_time
        selected_examples = []
        # 初始化一个空列表，用于存储和筛选后的级联数据
        for i in range(len(examples)):
            # 遍历所有级联数据
            example = examples[i]
            # print(f"example: {example}")
            # example：一个包含用户参与的级联列表(每个级联是用户的集合)
            example_times = examples_times[i]
            # print(f"example_times: {example_times}")
            # 每个级联中用户参与的时间戳列表
            if isinstance(example, list):
                # 如果example和example_list是python列表，则将他们转换成pytorch张量
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
                # torch.tensor:将列表转换为张量，以便进行更高效的数值计算
                # dtype=torch.float64:指定时间戳的张量为浮点型
            selected_example = torch.where((example_times < end_time) & (example_times >= start_time), example, torch.zeros_like(example))
            # 筛选逻辑：用户的时间戳example_times在当前时间段内 满足条件的为example，不满足生成一个与example形状相同的全零张量。
            # example_times是一个张量，与标量start_time和end_time的比较会逐元素进行
            # print(f"selected_example: {selected_example}")
            selected_examples.append(selected_example.numpy().tolist())
            # 将selected_example从张量转换为numpy数组，再转换为python列表
        sub_hypergraph = CascadeHypergraph(selected_examples, user_size, device=device)
        # 调用CascadeHypergragh函数，用当前时间段筛选后的用户集合构建超图
        # print(sub_hypergraph)
        hypergraph_list.append(sub_hypergraph)
        # 将构建好的子超图添加到Hypergragh_list列表中

    # =============== 最后一张超图 ===============
    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times >= start_time, example, torch.zeros_like(example))
        # print(selected_example)
        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(CascadeHypergraph(selected_examples, user_size, device=device))
    return hypergraph_list


'''
def DynamicCasHypergraph(examples, examples_times, user_size, device, step_split=8):
    hypergraph_list = []
    time_sorted = []
    print(examples_times)
    # 修正点：排除每个级联的最后一个时间戳（结束标记）
    for time in examples_times:
        time_sorted += time[:-1]  # 关键修改

    time_sorted = sorted(time_sorted)
    total_time_points = len(time_sorted)

    # 动态调整split_length防止除零错误
    split_length = max(total_time_points // step_split, 1)

    # 打印关键参数辅助调试
    print(f"总时间点: {total_time_points}, 分段长度: {split_length}")

    start_time = 0
    end_time = 0

    # 调整循环条件确保完全覆盖
    for x in range(split_length, split_length * (step_split + 1), split_length):
        x = min(x, total_time_points - 1)  # 防止索引越界
        start_time = end_time
        end_time = time_sorted[x]

        # 调试输出时间段信息
        print(f"划分时段[{len(hypergraph_list) + 1}]: {start_time:.2f} - {end_time:.2f}")

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i][:-1]  # 同步排除结束标记

            # 转换数据格式
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)

            # 精确筛选条件
            mask = (example_times >= start_time) & (example_times < end_time)
            selected_example = torch.where(mask, example, torch.zeros_like(example))

            # 收集有效用户
            valid_users = selected_example[selected_example.nonzero()].tolist()
            if valid_users:
                selected_examples.append(valid_users)

        # 构建超图前检查有效数据
        if selected_examples:
            hypergraph = CascadeHypergraph(selected_examples, user_size, device)
            print(f"生成超图: {hypergraph}")  # 调试输出
            hypergraph_list.append(hypergraph)

    return hypergraph_list
    '''




