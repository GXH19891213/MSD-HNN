import torch
import collections
import numpy as np
from torch.utils.data import Dataset, dataloader, random_split

class FakenewsDataset(Dataset):
    '''
    def __init__(self, walk_list, inner_list, type_list, label_list, graph, transform=False):
        
        self.labels = label_list
        # self.walk_list = np.array(walk_list, dtype=np.intc)
        self.walk_list = walk_list
        # self.inner_list = inner_list
        self.inner_list = [np.array(inner, dtype=np.intc) for inner in inner_list]
        self.graph = graph
        # self.type_list = np.array(type_list, dtype=np.intc)
        self.type_list = [np.array(type_, dtype=np.intc) for type_ in type_list]
        if transform:
            self.walk_list = torch.tensor(self.walk_list)
            self.inner_list = torch.tensor(self.inner_list)
            self.labels = torch.tensor(self.labels)
            self.type_list = torch.tensor(self.type_list)

        self.walk_num = self.walk_list.shape[0]
     '''

    def __init__(self, walk_list, inner_list, type_list, label_list, graph, transform=False):
        self.labels = label_list
        self.graph = graph

        # 将 walk_list、inner_list 和 type_list 转换为 NumPy 数组
        self.walk_list = np.array(walk_list, dtype=np.int32)  # 转为 NumPy 数组，确保形状一致
        self.inner_list = np.array([np.array(inner, dtype=np.intc) for inner in inner_list], dtype=np.int32)
        self.type_list = np.array([np.array(type_, dtype=np.intc) for type_ in type_list], dtype=np.int32)

        if transform:
            # 如果 transform=True，将数据转换为 PyTorch 张量
            self.walk_list = torch.tensor(self.walk_list, dtype=torch.int32)
            self.inner_list = torch.tensor(np.stack(self.inner_list), dtype=torch.int32)
            self.labels = torch.tensor(self.labels, dtype=torch.int32)
            self.type_list = torch.tensor(np.stack(self.type_list), dtype=torch.int32)
        # print("self.walk_list 类型：", type(self.walk_list))
        # print("self.walk_list dtype：", self.walk_list.dtype)
        # print("self.walk_list 内容：", self.walk_list)
        # np.stack(self.inner_list) 和 np.stack(self.type_list) 将两个列表的内嵌列表按轴堆叠成一个 NumPy 数组，再转换为张量。

        # walk_num 为随机游走的数量
        self.walk_num = len(self.walk_list)

    def __len__(self):
        return self.labels.shape[0]
    # 表示标签列表的第一个维度大小，通常表示样本数量
    def __getitem__(self, idx):
        return self.walk_list[idx], self.inner_list[idx], self.type_list[idx], self.labels[idx]
    # python中支持索引访问的特殊方法，可以使用dataset[idx]来访问数据

    def get_train_id(self):
        train_idx = np.load('./data/pili/train_id.npy')
        val_idx = np.load('./data/pili/val_id.npy')
        test_idx = np.load('./data/pili/test_id.npy')


        train_walks = []
        val_walks = []
        test_walks = []

        # 游走路径
        train_walk_lb = []
        val_walk_lb = []
        test_walk_lb = []

        # 游走路径标签
        train_inner_list = []
        val_inner_list = []
        test_inner_list = []

        # 内部列表信息
        train_type_list = []
        val_type_list = []
        test_type_list = []

        # 类型列表
        '''根据新闻节点的划分填充训练集和测试集'''
        for (w, lb, in_list, t_list) in zip(self.walk_list, self.labels, self.inner_list, self.type_list):
            if w[0] in train_idx:
                train_walks.append(w)
                train_walk_lb.append(lb)
                train_inner_list.append(in_list)
                train_type_list.append(t_list)
            elif w[0] in val_idx:
                val_walks.append(w)
                val_walk_lb.append(lb)
                val_inner_list.append(in_list)
                val_type_list.append(t_list)
            elif w[0] in test_idx:
                test_walks.append(w)
                test_walk_lb.append(lb)
                test_inner_list.append(in_list)
                test_type_list.append(t_list)


            # 创建对应的训练集、测试集和验证集数据集对象
        train_dataset = FakenewsDataset(train_walks, train_inner_list, train_type_list, train_walk_lb, self.graph,
                                        transform=True)
        val_dataset = FakenewsDataset(val_walks, val_inner_list, val_type_list, val_walk_lb, self.graph, transform=True)

        test_dataset = FakenewsDataset(test_walks, test_inner_list, test_type_list, test_walk_lb, self.graph,
                                       transform=True)

        return train_dataset, val_dataset, test_dataset

class Vocab:  
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  
    
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Data():
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.X = 0

        self.vocab = None
        self.corpus = None
        
        self.y = 0
        self.MAX_W_LEN = 10 
        
    def __len__(self):
        return self.x.size[0]
    
    def gen_dataset(self, walk_list, inner_list, type_list, labels, num_news, news_feature, entity_feature, topic_feature, test_ratio):
     
        self.train_data = Data("train Data")
        self.test_data = Data("test Data")
        self.vocab, self.corpus = self.gen_Vocab(walk_list, type_list)
       
        self.y = torch.tensor(labels, dtype=torch.float32)
        import random
        num_news = len(walk_list)
        indices = list(range(num_news))
        random.shuffle(indices)

        train_mask = torch.zeros(num_news)
        test_mask = torch.zeros(num_news)
        train_mask[indices[:round(num_news*test_ratio)+1]] = 1
        test_mask[indices[round(num_news*test_ratio)+1:]] = 1

        self.train_mask_news = train_mask.type(torch.BoolTensor)
        self.test_mask_news = test_mask.type(torch.BoolTensor)

        self.train_mask = torch.zeros(len(walk_list))
        self.test_mask = torch.zeros(len(walk_list))

        for i, walk in enumerate(walk_list):
            if self.train_mask_news[int(walk[0])] == True:
                self.train_mask[i] = 1
            else:
                self.test_mask[i] = 1
        self.train_mask = train_mask.type(torch.BoolTensor)
        self.test_mask = test_mask.type(torch.BoolTensor)
        walk_list = np.array(walk_list, dtype=np.int32)

        self.walk_list = torch.tensor(walk_list)
        self.inner_list = torch.tensor(np.array(inner_list,dtype=np.int))
        self.walk_types = np.array(type_list, dtype=np.string_)
        self.news_feature, self.entity_feature, self.topic_feature = news_feature, entity_feature, topic_feature

        self.train_data.walk_list = self.walk_list[self.train_mask]
        self.train_data.inner_list = self.inner_list[self.train_mask]
        self.train_data.type_list = self.walk_types[self.train_mask]
        self.train_data.news_feature, self.train_data.entity_feature, self.train_data.topic_feature = news_feature, entity_feature, topic_feature
        self.train_data.y = self.y[self.train_mask]
        self.test_data.walk_list = self.walk_list[self.test_mask]
        self.test_data.y = self.y[self.test_mask]
        self.test_data.type_list = self.walk_types[self.test_mask]
        return self.train_data, self.test_data

    def gen_Vocab(self, walk_list, type_list):
        corpus = []
        embeding_types = ["news", "topic", "entity"]
        for i, walk in enumerate(walk_list):
            t_list = type_list[i]
            for j, id in enumerate(walk):
                if t_list[j] not in embeding_types:
                    
                    corpus.append(id)
        vocab = Vocab(corpus)
        
        return vocab, corpus