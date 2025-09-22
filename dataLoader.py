"""
"""
import numpy as np
import pandas
import torch
from torch.autograd import Variable
import Constants
import pickle
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from build_graph import *


class Options(object):

    def __init__(self, data_name='poli'):
        self.news_centered = 'data/' + data_name + '/Processed/news_centered.pickle'
        self.user_centered = 'data/' + data_name + '/Processed/user_centered.pickle'
        # self.user_features = 'data/' + data_name + '/user_features.pickle'
        self.test_data = 'data/' + data_name + '/Processed/test_processed.pickle'
        self.valid_data = 'data/' + data_name + '/Processed/valid_processed.pickle'
        self.train_data = 'data/' + data_name + '/Processed/train_processed.pickle'
        self.news_features = 'data/' + data_name + '/struct_temp.pkl'
        # 存储新闻结构和时间特征，每条新闻的结构如子传播链数量和时间特征如传播持续时间
        self.news_mapping = 'data/' + data_name + '/news_mapping.pickle'
        # 将新闻ID映射到一个索引中
        self.save_path = ''


def DataReader(data_name):
    # 从options配置中读取训练、验证和测试数据文件路径，使用pickle将这些文件加载为python对象
    options = Options(data_name)
    with open(options.train_data, 'rb') as f:
        train_data = pickle.load(f)
    with open(options.valid_data, 'rb') as f:
        valid_data = pickle.load(f)
    with open(options.test_data, 'rb') as f:
        test_data = pickle.load(f)

    # print(train_data)

    total_size = len(train_data[0]) + len(test_data[0]) + len(valid_data[0])
    train_size = len(train_data[0])
    valid_size = len(valid_data[0])
    test_size = len(test_data[0])
    print("news cascades size:%d " % (total_size))
    print("train size:%d " % (len(train_data[0])))
    print("test and valid size:%d " % (len(test_data[0]) + len(valid_data[0])))
    return train_data, valid_data, test_data, total_size, train_size, valid_size, test_size
    # 计算并打印数据集的总大小、训练集、测试集和验证集。


def FeatureReader(data_name):
    options = Options(data_name)
    with open(options.news_mapping, 'rb') as handle:
        n2idx = pickle.load(handle)
    with open(options.news_features, 'rb') as f:
        '''Spread status: S1, S2, T1, T2 
            Structural：(S1)number of sub-cascades, (S2)proportion of non-isolated cascades;
            Temporal:  (T1) duration of spread,(T2) the average response time from tweet to retweet'''
        # 传播状态：S1、S2、T1、T2
        # 结构方面：（S1）子级联的数量，（S2）非孤立级联的比例；
        # 时间方面：（T1）传播持续时间，（T2）从推文到转发的平均响应时间
        # options.news_features:
        # Data type: <class 'numpy.ndarray'>
        # Length of list: 314
        # Item 0: ['politifact14040' '25.0' '0.2' '3867804' '75718']
        # Item 1: ['politifact13731' '170.0' '0.06470588235294118' '18948155' '47133']
        features = np.array(pickle.load(f))
        news_size = len(features)
        spread_status = np.zeros((news_size + 1, 4))
        # 初始化一个形状为(news_size+1,4)的全零数组，用于存储传播状态特征，+1可能是为了预留弟0行作为占位符
        for news in features:
            # print(news)
            spread_status[n2idx[news[0]]] = np.array(news[1:])
            # 预留第一行，将特征数据填充到 spread_status 数组中相应的行
            # print(spread_status[n2idx[news[0]]])
    return spread_status


def GraphReader(data_name):
    options = Options(data_name)
    with open(options.news_centered, 'rb') as f:
        news_centered_graph = pickle.load(f)
    with open(options.user_centered, 'rb') as f:
        user_centered_graph = pickle.load(f)
    # news_centered和user_centered文件分别包含新闻传播的序列和用户行为序列
    useq, user_inf = (item for item in user_centered_graph)
    # print(f"!!!!user_inf:{user_inf}")
    # useq：用户行为序列（如用户的传播参与记录）
    # user_inf:用户影响力信息
    seq, timestamps, user_level, news_inf = (item for item in news_centered_graph)
    # 新闻传播的用户序列 时间戳 用户层级信息 新闻其他信息
    spread_status = FeatureReader(data_name)
    # 调用 FeatureReader 函数，读取新闻的传播状态特征（如子传播链数量、非孤立传播链比例等
    user_size = len(useq)
    # 初始化一个空列表来存储所有整数
    all_integers = []

    user_inf[user_inf > 0] = 1
    '''
    for sublist in user_inf:
        for i in range(len(sublist)):
            if sublist[i] > 0:
                sublist[i] = 1
    '''
    # 将user_inf数组中所有大于0的元素设置为1
    # 二值化用户影响力数据，将其转换为是否活跃的二元状态
    # act_level = user_inf[1:].sum(1)
    act_level = user_inf.sum(1)
    # act_level = [sum(sublist) for sublist in user_inf[1:]]
    # 计算每个用户的活跃度
    # user_inf[1:]:跳过第0行，占位符
    # .sum(1)：按行求和，计算每个用户的活跃总数
    #avg_inf = np.append([0], act_level) 这里是因为报错user_cen长度比user_size多1，改掉了
    avg_inf = act_level
    # print(f"avg_inf:{avg_inf.shape}")
    # 使用np.append在活跃度数据前添加一个0(占位符)，
    # 生成avg_inf,为了确保每个用户的活跃度有一个对应的位置
    news_centered_graph = [seq, timestamps, user_level]
    user_centered_graph = [useq, news_inf, avg_inf]
    return [torch.LongTensor(i).to(Constants.device) for i in news_centered_graph], \
        [torch.LongTensor(i).to(Constants.device) for i in user_centered_graph], \
        torch.LongTensor(spread_status).to(Constants.device)



class DataLoader(object):
    ''' For data iteration '''
    # 定义一个数据加载器类，用于处理数据的批次迭代，
    # DataLoader用于将数据集分批次地提供给模型进行训练或者测试
    def __init__(self, data, batch_size=32, cuda=True, test=False):
        # 我从32改到64
        # test:指示是否处于测试模式，影响数据处理方式
        self._batch_size = batch_size
        self.idx = data[0]  # data[0]是数据集的输入，新闻索引
        self.label = data[1]  # data[1]是数据标签
        self.test = test  # 是否是测试模型，如果是，将不会计算梯度
        self.cuda = cuda  # 是否使用GPU加速

        self._n_batch = int(np.ceil(len(self.idx) / self._batch_size))
        # 计算总批次数量 self._n_batch np.ceil()向上取整
        # print(self._n_batch)
        # train:2个批次 验证集:1 test:7
        self._iter_count = 0
        # 跟踪当前批次的索引

    def __iter__(self):
        return self

    # __iter__()方法返回迭代器本身，
    # 任何实现了__iter()__方法的对象都可以视为迭代器，这个方法不需要接受任何参数，返回值为迭代器对象本身
    def __next__(self):
        return self.next()

    # __next__()方法用于返回迭代器的下一个元素，当迭代器被请求下一个元素时，
    # __next__()方法会被调用。如果还有元素可以返回，__next__()应该返回下一个元素；
    # 如果没有更多的元素，应该抛出StopIteration异常来表示迭代结束
    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        # 获取下一个批次的数据
        def seq_to_tensor(insts):
            # 定义一个将输入数据序列转换成PyTorch张量(Tensor)的函数，
            # insts:输入数据序列，可以是列表或者数组
            inst_data_tensor = Variable(
               torch.LongTensor(insts), volatile=self.test)
            #   torch.LongTensor(inst):输入序列insts转换成Pytorch的LongTensor类型
            #  旧版本的Pytorch中，variable用于封装张量，并允许自动求导(计算梯度) pytorch4.0，
            #  variable被合并到torch.tensor中，直接使用张量
            #  volatile=True 表示张量不需要计算梯度，用于推理模式（测试时不需要反向传播）
            return inst_data_tensor
            # 返回转换后的PyTorch张量，用于神经网络的输入

        if self._iter_count < self._n_batch:
            # self._iter_count:当前已处理的批次数量
            # self._n_batch：总批次数量
            # 检测是否还有剩余批次
            batch_idx = self._iter_count
            # batch_idx：保存当前批次索引。每处理一个批次就将self._iter_count增加1
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            # 当前批次的起始位置
            end_idx = (batch_idx + 1) * self._batch_size
            # end_idx:当前批次的结束位置，但是不包含该位置的索引
            idx = self.idx[start_idx:end_idx]
            # self.idx:输入数据，如新闻或者用户索引序列
            labels = self.label[start_idx:end_idx]
            # 当前批次对应标签数据
            idx = seq_to_tensor(idx)
            labels = seq_to_tensor(labels)
            # 将输入数据和标签转换为PyTorch的LongTensor类型，
            # 根据是否测试模式设置volatile标志

            return idx, labels

        else:

            self._iter_count = 0
            raise StopIteration()
            # 所有批次数据处理完毕，重置self._iter_count为0，抛出StopIteration异常

'''
class GraphDataloader:
    # For graph data iteration ensuring consistency across train/val/teat batches
    def __init__(self, data, graphdata, batch_size=32, cuda=True, test=False):
        self.batch_size = batch_size
        self.cuda = cuda
        self.test = test
        self.graphdata = graphdata
        # Load the labels
        self.idx = data[0]  # data[0]是数据集的输入，新闻索引
        self.label = data[1]  # data[1]是数据标签
        self._n_batch = int(np.ceil(len(self.idx) / self.batch_size))
        # print(f"!!!!!!!self._n_batch: {self._n_batch}") 三次输出 2 1 7
        self._iter_count = 0
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def next(self):
        #  Get the next batch for training/validation/testing 
        def seq_to_tensor(insts):
            inst_data_tensor = Variable(torch.LongTensor(insts), volatile=self.test)
            return inst_data_tensor
        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            # Calculate the start and end index of the current batch
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx+1)*self.batch_size

            # Get the indices and labels for the current batch
            idx = self.idx[start_idx:end_idx]
            labels = self.label[start_idx:end_idx]

            # Convert idx and labels to tensors
            idx_tensor = seq_to_tensor(idx)
            labels_tensor = seq_to_tensor(labels)
            # 获取当前批次的新闻索引
            news_indices = torch.tensor(idx, dtype=torch.long)
            # Use NeighborLoader to sample the neighbors for the news nodes
            loader = NeighborLoader(self.graphdata, num_neighbors=[2,2], batch_size=self.batch_size, shuffle=False, input_nodes=('news', news_indices))
            # input_nodes = ['news'] 指定了图数据值哪些类型的节点将作为输入来进行邻居采样
            # 这里实际上num_neighbors=[10,10]相当大了，一次就可能把所有节点采完了
            batch = next(iter(loader))
            # return the batch for training
            #return batch['news'].x, labels_tensor
            print(f"batch:{batch}")
            # print({f"labels_tensor:{labels_tensor}"})
            return  batch, labels_tensor, news_indices
        else:
            raise StopIteration()
    def __len__(self):
        # return the total number of batches
        return self._n_batch
    '''
import torch
from torch.autograd import Variable
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData

import torch
from torch.autograd import Variable
from torch_geometric.loader import NeighborLoader
import argparse
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

# 原有的 build_graph、class2global、get_edgeList、load_graph 函数保持不变

class GraphDataloader:
    '''For graph data iteration ensuring consistency across train/val/test batches'''
    def __init__(self, data, graphdata, batch_size=32, cuda=True, test=False):
        self.batch_size = batch_size
        self.cuda = cuda
        self.test = test
        self.graphdata = graphdata
        # Load the labels
        self.idx = data[0]  # data[0]是数据集的输入，新闻索引
        self.label = data[1]  # data[1]是数据标签
        self._n_batch = int(np.ceil(len(self.idx) / self.batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ''' Get the next batch for training/validation/testing '''
        def seq_to_tensor(insts):
            inst_data_tensor = Variable(torch.LongTensor(insts), volatile=self.test)
            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            # Calculate the start and end index of the current batch
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size

            # Get the indices and labels for the current batch
            idx = self.idx[start_idx:end_idx]
            labels = self.label[start_idx:end_idx]

            # Convert idx and labels to tensors
            idx_tensor = seq_to_tensor(idx)
            labels_tensor = seq_to_tensor(labels)
            # 获取当前批次的新闻索引
            news_indices = torch.tensor(idx, dtype=torch.long)
            # Use NeighborLoader to sample the neighbors for the news nodes
            loader = NeighborLoader(self.graphdata, num_neighbors=[5, 5], batch_size=self.batch_size, shuffle=False,
                                    input_nodes=('news', news_indices))
            # num_neighbors=[5,5]两层邻居采样，每层最多5个邻居
            # input_nodes=('news', news_indices)：指定起始节点类型和索引。

            batch = next(iter(loader))
            existing_news_indices = set()
            # 初始化空集合，存储批次图中实际出现的新闻节点索引

            existing_news_indices = set(batch['news'].n_id.cpu().numpy())  # 直接使用 n_id
            existing_news_indices = torch.tensor(list(existing_news_indices), dtype=torch.long)
            # print(f"batch['news'].n_id:{batch['news'].n_id}")
            # print(f"existing_news_indices:{existing_news_indices}")
            # 将集合转为列表，创建Pytorch张量

            # 对比索引
            missing_indices = []
            for idx in news_indices:
                if idx not in existing_news_indices:
                    missing_indices.append(idx)

            if missing_indices:
                missing_indices = torch.tensor(missing_indices, dtype=torch.long)
                print("Missing global indices:", missing_indices)

                # 添加缺失的节点特征
                missing_features = self.graphdata['news'].x[missing_indices]
                new_x = torch.cat([batch['news'].x, missing_features], dim=0)
                batch['news'].x = new_x

                # 更新 n_id
                if 'n_id' in batch['news']:
                    batch['news'].n_id = torch.cat([batch['news'].n_id, missing_indices], dim=0)
                else:
                    batch['news'].n_id = missing_indices
                # print("Updated n_id for news:", batch['news'].n_id)

                # 全局到局部映射（针对所有节点类型）
                news_global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in
                                        enumerate(batch['news'].n_id)}
                entity_global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in
                                          enumerate(batch['entity'].n_id)}
                topic_global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in
                                         enumerate(batch['topic'].n_id)}
                # print("News global to local mapping:", news_global_to_local)
                # print("Entity global to local mapping:", entity_global_to_local)
                # print("Topic global to local mapping:", topic_global_to_local)

                # 添加缺失节点的边（只添加源节点为 missing_indices 的边）
                for edge_type in self.graphdata.edge_types:
                    src_type, rel, dst_type = edge_type
                    if src_type == 'news':  # 只处理源节点为 'news'
                        edge_index = self.graphdata[edge_type].edge_index
                        mask = torch.isin(edge_index[0], missing_indices)  # 只检查源节点
                        new_edges_global = edge_index[:, mask]
                        # 提取源节点在 missing_indices 中的边,如：tensor([[313], [20]]
                        # print(f"Global edges for {edge_type}:\n", new_edges_global)

                        # 转换为局部索引
                        new_edges_local = torch.zeros_like(new_edges_global)
                        for i in range(new_edges_global.shape[1]):
                            # 遍历每条边的列索引
                            src_global = new_edges_global[0, i].item()
                            # 访问第i列源索引
                            dst_global = new_edges_global[1, i].item()
                            # 访问第i列目标索引
                            # 源节点（news）映射到局部索引
                            new_edges_local[0, i] = news_global_to_local[src_global]
                            # 目标节点根据类型映射到局部索引
                            if dst_type == 'entity':
                                new_edges_local[1, i] = entity_global_to_local[dst_global]
                            elif dst_type == 'topic':
                                new_edges_local[1, i] = topic_global_to_local[dst_global]
                            elif dst_type == 'news':
                                new_edges_local[1, i] = news_global_to_local[dst_global]
                            else:
                                raise ValueError(f"Unknown dst_type: {dst_type}")
                        # print(f"Local edges for {edge_type}:\n", new_edges_local)

                        # 更新批次中的边索引
                        if 'edge_index' not in batch[edge_type]:
                            batch[edge_type].edge_index = new_edges_local
                        else:
                            batch[edge_type].edge_index = torch.cat([batch[edge_type].edge_index, new_edges_local],
                                                                    dim=1)
            return batch, labels_tensor, news_indices
        else:
            raise StopIteration()

    def __len__(self):
        # return the total number of batches
        return self._n_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='choose dataset & hiddenSize')
    parser.add_argument('--dataset', type=str, default='poli')
    parser.add_argument('--hiddenSize', type=int, default=600, help="news_emb_size")

    args = parser.parse_args()
    dataset = args.dataset
    hiddenSize = args.hiddenSize

    data_sum_graph = build_graph(dataset, hiddenSize)
    edgeList_rw = get_edgeList(dataset, hiddenSize)

    # 示例数据
    data = (np.arange(data_sum_graph['news'].x.shape[0]), np.random.randint(0, 2, data_sum_graph['news'].x.shape[0]))
    dataloader = GraphDataloader(data, data_sum_graph, batch_size=32)

    for batch, labels_tensor, news_indices in dataloader:
        pass

    print(f'graph & edgelist for {dataset} done')