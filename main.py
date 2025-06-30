import argparse
import torch
from torch_geometric import data
# 用于图神经网络的工具库
from torch.utils.data import Dataset, random_split
from dataLoader import DataReader,  DataLoader
from model.model1 import Model1, train
import json
from Data1 import Vocab, Data, FakenewsDataset
import news_RandomWalk as rw
from utils import load_data
# utils工具函数模块，包括load_data函数

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# 定义空的warn函数，替代warnings.warn。导入warning模块，覆盖其警告功能，在运行程序时屏蔽所有警告，保证输出简洁
class Config():
    def __init__(self):
        self.name = "model config"
    
    def print_config(self):
        for attr in self.attribute:
            print(attr)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='poli')
    parser.add_argument('--hiddenSize', type=int, default=600)
    parser.add_argument('--config', type=str, default='./config/HeteroSGT.json', help='configuration file name.')
    parser.add_argument('--model', type=str, default='Model1')
    parser.add_argument('--num_laps', type=int, default=1, help="num_laps")
    # 遍历图结构的次数
    parser.add_argument('--walk_length', type=int, default=5, help="walk length")
    parser.add_argument('--round', type=int, default=1, help='test round')
    parser.add_argument('--num_layers', type=int, default=5, help="num_layers")
    parser.add_argument('--case2', type=str, default="no", help="yes or no for case study II")
    parser.add_argument('--case3', type=str, default="h0", help="h0, mean, or max for case study III")
    parser.add_argument('--restart', type=float, default=0.1, help=" probability of restarts in rw")
    parser.add_argument('--topn', type=int, default=3, help=" num_topics each news linked to")
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")
    # 设置每个新闻连接的主题数量
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 检查是否有可以GPU，如果有设置为CUDA,如果没有，使用CPU
    args.device = device
    print(f"Training Device: {device}, Dataset: {args.dataset},  Model : {args.model}, Test Round: {args.round}, num_laps: {args.num_laps}, walk_length: {args.walk_length}, hiddenSize: {args.hiddenSize},num_layers: {args.num_layers}")

    with open(args.config, 'r') as f:
        # 打开命令行中指定的JSON配置文件
        # 通过json.load将json文件解析为字典
        config_dicts = json.load(f)
    configs = {}
    for config in config_dicts:
        conf = Config()
        # 遍历解析后的配置文件，将每一项动态加载到config类的实例
        for key, value in config.items():
            setattr(conf, key, value)
            # 动态地将JSON文件中的每一项配置添加到Config类的实例中
        configs.update({
            config["dataset"] : conf
        })  # 将配置按照数据集为键存储到config字典
    config = configs[args.dataset]
    # 根据命令行中指定的数据集名称，选择对应的配置对象
    

    graph, train_data, val_data, test_data  = load_data(args)
    # 调用load_data函数加载图数据集和训练、测试数据
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, cuda=False)
    valid_dataloader = DataLoader(val_data, batch_size=opt.batch_size, cuda=False)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, cuda=False)

    # 使用DataLoader加载数据和测试数据，DataLoader用于将数据分批，并在训练过程中自动加载数据
    # 参数是 dataset、batch_size、shuffle

    # batch_size=train_data._len_():表示每个批次加载整个数据集 shuffle=True 随机打乱数据
    config.query_size, config.key_size, config.value_size = args.hiddenSize, args.hiddenSize, args.hiddenSize
    # 注意力机制中query、key、value的维度
    config.norm_shape = (args.walk_length, config.num_hiddens)
    # norm_size:归一化层的形状 walk_length:表示在进行图数据的随机游走时路径长度
    config.news_size =graph["news"].x.shape[0]
    # news_size：表示从graph中提取news节点的数量
    if graph["entity"] != {}:
        config.entity_size = graph["entity"].x.shape[0]
    else:
        config.entity_size = 0 
    if graph["topic"] != {}:
        config.topic_size = graph["topic"].x.shape[0]
    else:
        config.topic_size = 0
    # 如果graph中存在entity或topic节点，则记录其数量，否则设置为0
    config.walk_length = args.walk_length
    # 随机游走路径长度
    config.num_hiddens = args.hiddenSize
    # 隐藏层大小
    args.dropout = config.dropout
    train_label = train_data.labels
    test_label = test_data.labels
    model = globals()[args.model](config)
    # globals():返回当前全局命名空间中所有变量和对象。是一个字典，键为变量或对象名，值是对应变量或对象本身
    # global()[args.model]:在全局命名空间中，根据args.model的值找到对应的类或函数
    # globals()[args.model](config)：使用括号调用返回的函数，将config当做初始化参数传递 实例化
    model = model.double()
    # 将模型中所有参数和计算中数据类型设置为 torch.float64
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # 使用Adam优化器，lr=config.learning_rate:学习率
    # weight_decay=config.weight_decay:L2正则化系数
    train_label = train_data.labels
    test_label = test_data.labels
    train(model, train_data, train_label, test_data, test_label, args.epochs, optimizer, device, args)
    # model:初始化后的模型，train_data,test_data:训练集和测试集

