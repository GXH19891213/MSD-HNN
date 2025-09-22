import copy
from random import shuffle
# 用于随机打乱列表
from deepwalk import graph
import random
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm


def remove_dups(data):
    df = pd.DataFrame(data).astype(int)
    dup_index = df[df.duplicated(subset=df.columns)].index.values.tolist()
    df.drop_duplicates(subset=df.columns, inplace=True, ignore_index=True)
    newlist = df.values.tolist()
    num_dups = len(data) - len(newlist)
    print('Dups removed', '\n' + 'Num_Dups:', num_dups)
    if num_dups != 0:
        print('Dup_Index:', dup_index)
    return newlist, dup_index
# 该函数用于移除二维列表中的重复项,返回去重后的newslist和重复项的索引dup_index

def clean_list(data, dup_index):
    newlist = copy.deepcopy(data)
    for i, j in enumerate(dup_index):
        del newlist[j - i]
    return newlist
# 根据dup_list删除列表中对应的重复项

def global2df(data, colsname):
    # new_df = pd.DataFrame(list(data.items())[:len(list(data.items())) // 2], columns=colsname)
    new_df = pd.DataFrame(list(data.items()), columns=colsname)
    return new_df
# 将字典转换成DataFrame


def inner2df(data, colsname, typename):
    # new_df = pd.DataFrame(list(data.items())[:len(list(data.items())) // 2], columns=colsname)
    new_df = pd.DataFrame(list(data.items()), columns=colsname)
    new_df['type'] = [typename] * len(new_df)
    return new_df
# 于global2df类似，不过害添加了一个新的列'type'，表示数据的类型


def get_inner_type(dataset, walk_list):
    # 根据随机游走生成的路径列表walk_list，获取每个节点的内部ID和类型。
    news_index = np.load(f"./Data/{dataset}/graph/nodes/news_index.npy", allow_pickle=True).item()
    entity_index = np.load(f"./Data/{dataset}/graph/nodes/entity_index.npy", allow_pickle=True).item()
    topic_index = np.load(f"./Data/{dataset}/graph/nodes/topic_index.npy", allow_pickle=True).item()

    global_index1 = np.load(f"./Data/{dataset}/graph/nodes/global_index_graph1.npy", allow_pickle=True).item()

    global_df1 = global2df(global_index1, ["name", "g_id"])
    # 使用global2df函数将global_index1转换为一个DataFrame
    news_df = inner2df(news_index, ["name", "inner_id"], 0)
    entity_df = inner2df(entity_index, ["name", "inner_id"], 1)
    topic_df = inner2df(topic_index, ["name", "inner_id"], 2)
    inner_df1 = pd.concat([news_df, entity_df, topic_df], ignore_index=True)
    # 按行拼接，
    final_df = pd.merge(global_df1, inner_df1)
    # 将global_df1和inner_df1按节点名称name合并 合并全局索引和内部索引的DataFrame
    '''
    print("global_df1:\n", global_df1.head())
    print("inner_df1:\n", inner_df1.head())
    print("final_df:\n", final_df.head())
    '''
    inner_list = []  # 节点内部ID
    type_list = []  # 类型
    for walk in tqdm(walk_list, desc="getting inner_list & type_list ..."):
        inners = []
        types = []
        for j in walk:
            item = final_df[final_df.g_id == int(j)]
            inner = item['inner_id'].values.item()
            type_n = item['type'].values.item()
            inners.append(inner)
            types.append(type_n)
        inner_list.append(inners)
        type_list.append(types)
    return inner_list, type_list
    # 遍历 walk_list，获取每个节点的内部ID和类型，并存储在 inner_list 和 type_list 中。


def rand_walk(dataset, restart, num_laps=1, walk_length=5):
    # 加载图数据，从指定路径加载图的边列表，并将其转换为无向图
    '''
    dataset:数据集
    restart：重启概率
    num_laps：随机游走次数
    walk_length: 每次游走的步数
    '''
    G = graph.load_edgelist(f"./Data/{dataset}/graph/edges/{dataset}.edgelist", undirected=True)
    # 加载图的数据结构，这里加载的是边列表，用来表示图的结构
    # undirected=True 指明是无向图
    df = pd.read_excel(f"./Data/{dataset}/news_final.xlsx")
    # 包含news_id label两列
    num_news = len(df['news_id'].tolist())
    # 将news_id转换为列表格式，然后使用len()获取新闻数量，存储在num_news变量中
    label = df['label'].tolist()
    labels = label * num_laps
    print('num_laps:', num_laps, 'walk_length:', walk_length, 'num_news:', num_news)

    walk_list = []
    # 初始化空列表，用了存储所有新闻节点的游走路径
    for i in tqdm(range(num_laps), desc='news random walk...'):
        # 外部循环，进行num_laps次的随机游走
        for j in range(num_news):
            # 内层循环，对每一个新闻节点进行随机游走
            # walk = G.random_walk(j, walk_length, alpha=restart, rand=random.Random())
            walk = G.random_walk(path_length=walk_length, alpha=restart, rand=random.Random(), start=j)
            # 返回一个游走路径，包含多个节点
            walk_list.append(walk)
            # 将当前游走路径walk添加到walk_list中

    walk_list_, dup_index = remove_dups(walk_list)
    # 去除walk_list中重复路径
    labels_ = clean_list(labels, dup_index)
    inner_list, type_list = get_inner_type(dataset, walk_list)
    return walk_list_, labels_, inner_list, type_list



