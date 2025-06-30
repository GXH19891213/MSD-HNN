import pandas as pd
# 用于数据处理和加载Excel和CSV文件
import torch
# 是PyTorch库，提供张量操作和神经网络相关的功能
import numpy as np
# 数组处理，加载npy文件pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_cluster-1.6.0-cp38-cp38-win_amd64.whl
# pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_sparse-0.6.13-cp38-cp38-win_amd64.whl
# pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu115/torch_spline_conv-1.2.1-cp38-cp38-win_amd64.whl
from torch_geometric.data import HeteroData
# 用于构建异构图的数据集对象
import argparse
import os

'''加载边'''
def load_edge(dataset, node):
    # 确定文件格式，根据节点类型加载边信息
    if node == 'news':
        df = pd.read_csv(f'./data/{dataset}/graph/edges/news2news.csv', sep=',', encoding='utf-8')
    else:
        df = pd.read_excel(f'./data/{dataset}/graph/edges/news2{node}.xlsx')

    # 将 DataFrame 转换为列表，每一项是一个边的节点对
    pair = df.values.tolist()

    # 加载索引文件 .item()转换为字典
    news_index = np.load(f'./data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()

    # 构造反向映射：从本地索引值（整数）到字符串键新闻原始ID
    reverse_news_index = {v: k for k, v in news_index.items()}

    # 调试信息
    print(f"reverse_news_index sample: {list(reverse_news_index.items())[:10]}")

    index_dict = np.load(f'./data/{dataset}/graph/nodes/{node}_index.npy', allow_pickle=True).item()
    # 加载实体和主题的索引映射
    edges = []  # 正向边
    edges_ = []  # 反向边

    for i in pair:
        try:
            # 读取起点和终点
            head = news_index[i[0]]
            tail = index_dict[str(i[1])]
            edge = [head, tail]
            edge_ = [tail, head]
            # 遍历pair中每一对节点，将其映射到对应索引
            # 正向边和反向边分别被存储在edges和edges_列表
            edges.append(edge)
            edges_.append(edge_)

        except KeyError as e:
            print(f"KeyError: {e}. Available keys in index_dict: {list(index_dict.keys())[:10]}")
            print(f"Problematic pair: {i}")
            continue  # 跳过出错的边

    return edges, edges_
    # 返回正向边和反向边列表

'''构建异质图'''
def build_graph(dataset,hiddenSize):
    news_attr = np.load(f'./data/{dataset}/graph/nodes/news_embeddings_{hiddenSize}_final.npy')
    news_attr = torch.from_numpy(news_attr).float()  # 转换为 float32
    # 加载新闻节点的嵌入向量
    entity_attr = np.load(f'./data/{dataset}/graph/nodes/entity_embeddings_{hiddenSize}.npy')
    entity_attr = torch.from_numpy(entity_attr).float()  # 转换为 float32
    # 加载实体节点的嵌入向量
    topic_attr = np.load(f'./data/{dataset}/graph/nodes/topic_embeddings_{hiddenSize}.npy')
    topic_attr = torch.from_numpy(topic_attr).float()  # 转换为 float32
    # 主题节点嵌入向量
    news2entity, news2entity_ = load_edge(dataset,'entity')
    print(f"news2entity edges: {len(news2entity)}")  # 打印正向边数量
    print(f"news2entity_ (reversed) edges: {len(news2entity_)}")  # 打印反向边数量
    news2topic, news2topic_ = load_edge(dataset,'topic')
    print(f"news2topic edges: {len(news2topic)}")
    print(f"news2topic_ (reversed) edges: {len(news2topic_)}")
    news2news, news2news_ = load_edge(dataset,'news')
    print(f"news2news edges: {len(news2news)}")
    print(f"news2news_ (reversed) edges: {len(news2news_)}")
    # 加载新闻与实体、 新闻与主题、 新闻与新闻之间的边
    df_news = pd.read_excel(f'./Data/{dataset}/news_final.xlsx')
    # 从指定路径读取 [新闻ID,0/1] 保存为pandas的DataFrame对象
    label = df_news['label'].tolist()
    # 保存标签信息
    data = HeteroData()
    # 创建HeteroData对象，存储异质图数据
    data['news'].x = news_attr
    data['entity'].x = entity_attr
    data['topic'].x = topic_attr
    # 将新闻、实体、主题节点的特征分别存储在data对象中'news'、'entity'和'topic'字典下
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity, dtype=torch.long).t().contiguous()
    data['entity', 'rev_has', 'news'].edge_index = torch.tensor(news2entity_, dtype=torch.long).t().contiguous()
    # 将新闻和实体之间的边存储到data对象，边的索引存储在edge_index中， .t().contiguous()是为了将边索引转置为正确形状
    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    data['topic', 'rev_belongs', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()

    data['news', 'links', 'news'].edge_index = torch.tensor(news2news, dtype=torch.long).t().contiguous()
    data['news', 'rev_links', 'news'].edge_index = torch.tensor(news2news_, dtype=torch.long).t().contiguous()

    data['news'].y = torch.tensor(label, dtype = torch.long)
    # 将新闻节点的标签（目标）存储到 data 对象中的 'news' 节点

    print('='*60)
    print('HeteroGraph:', dataset, '\n', data)
    print(' num_nodes:', data.num_nodes, '\n', 'num_edges:', data.num_edges, '\n', 'Data has isolated nodes:', data.has_isolated_nodes(), '\n', 'Data is undirected:', data.is_undirected())
    print('='*60, '\n')
    torch.save(data, f'./data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt')
    return data

'''将边列表中本地索引转换为全局索引'''
def class2global(edgelist, global_index, classindex):
    print(f"edgelist: {edgelist[:10]}")  # 打印 edgelist 样例
    print(f"global_index keys (sample): {list(global_index.keys())[:10]}")  # 打印 global_index 的键样例
    print(f"global_index values (sample): {list(global_index.values())[:10]}")  # 打印 global_index 的值样例
    print(f"classindex keys (sample): {list(classindex.keys())[:10]}")  # 打印 classindex 的键样例
    print(f"classindex values (sample): {list(classindex.values())[:10]}")  # 打印 classindex 的值样例

    # 创建 classindex 的反向映射字典，将本地类索引映射回类名
    reverse_classindex = {v: k for k, v in classindex.items()}
    print(f"reverse_classindex sample: {list(reverse_classindex.items())[:10]}")  # 调试信息

    indices_g = []
    for i in edgelist:
        try:
            # 从反向映射获取字符串键
            ID = reverse_classindex[i]  # 如 2 -> 'politifact6657'
            #  print(f"Mapped i ({i}) to ID: {ID}")

            # 从 global_index 获取对应的全局索引
            index_g = global_index[ID]
             # print(f"Mapped ID ({ID}) to global index: {index_g}")

            indices_g.append(index_g)
        except KeyError as e:
            print(f"KeyError: {e}. i={i}, reverse_classindex keys: {list(reverse_classindex.keys())[:10]}")
            raise
    print(f"indices_g (sample): {indices_g[:10]}")  # 打印最终生成的索引列表
    return indices_g

'''获取图的边列表，获取图的边列表，将边列表保存到edgelist文件'''
def get_edgeList(dataset, hiddenSize):
    # 加载本地索引文件
    news_index = np.load(f'./data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'./data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'./data/{dataset}/graph/nodes/topic_index.npy', allow_pickle=True).item()

    # 加载全局索引
    global_index = np.load(f'./data/{dataset}/graph/nodes/global_index_graph1.npy', allow_pickle=True).item()

    # 加载图数据，返回一个图对象data
    data = torch.load(f'./data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt')

    # 删除反向边
    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']
    del data['news', 'links_', 'news']

    # 提取边索引
    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    # 新闻节点索引
    entityList = data['news', 'has', 'entity'].edge_index.tolist()[1]
    newsList1 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]
    news_List_h = data['news', 'links', 'news'].edge_index.tolist()[0]
    news_List_t = data['news', 'links', 'news'].edge_index.tolist()[1]
    # edge_index是形状为(2, num_edges)的张量。第一行表示起点节点索引 ，第二行表示终点索引
    # 调试输出
    print(f"newsList0 sample: {newsList0[:10]}")
    print(f"entityList sample: {entityList[:10]}")

    # 调用 class2global,将每个边列表本地索引转换为全局索引
    news0_g = class2global(newsList0, global_index, news_index)
    entity_g = class2global(entityList, global_index, entity_index)
    news1_g = class2global(newsList1, global_index, news_index)
    topic_g = class2global(topicList, global_index, topic_index)
    news_h_g = class2global(news_List_h, global_index, news_index)
    news_t_g = class2global(news_List_t, global_index, news_index)

    # 合并全局边
    node_head = news0_g + news1_g + news_h_g
    node_tail = entity_g + topic_g + news_t_g
    edgeList_rw = [f"{head} {tail}" for head, tail in zip(node_head, node_tail)]

    # 保存到文件
    with open(f'./data/{dataset}/graph/edges/{dataset}.edgelist', 'w', encoding='utf-8') as f:
        f.writelines([edge + '\n' for edge in edgeList_rw])

    return edgeList_rw
import torch

def load_graph(dataset, hiddenSize):
    # 定义文件路径
    file_path = f'./data/{dataset}/graph/{dataset}_{hiddenSize}_final.pt'

    # 使用 torch.load 加载数据
    data = torch.load(file_path)

    # 输出加载的数据
    print('=' * 60)
    print('Loaded HeteroGraph:', dataset, '\n', data)
    print('num_nodes:', data.num_nodes, '\n', 'num_edges:', data.num_edges, '\n',
          'Data has isolated nodes:', data.has_isolated_nodes(), '\n', 'Data is undirected:', data.is_undirected())
    print('=' * 60, '\n')

    return data



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='choose dataset & hiddenSize')
    parser.add_argument('--dataset', type=str, default='poli')
    parser.add_argument('--hiddenSize', type=int, default=600, help="news_emb_size")

    args = parser.parse_args()
    dataset = args.dataset
    hiddenSize = args.hiddenSize
    
    data_sum_graph = build_graph(dataset,hiddenSize)
    edgeList_rw = get_edgeList(dataset,hiddenSize)
    
    print(f'graph & edgelist for {dataset} done') 
