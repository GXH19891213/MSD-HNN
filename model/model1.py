import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from utils import test_once, print_results_once, save_results
from model.Base import TransformerEncoder, FNclf, genX
import os

class Model1(nn.Module):
    def __init__(self, config, **kwargs) -> None:
      
        super(Model1, self).__init__(**kwargs)
    
        key_size = config.key_size
        query_size = config.query_size
        value_size = config.value_size
        num_hiddens = config.num_hiddens
        # 隐藏层维度
        norm_shape = config.norm_shape
        # 归一化层的输入形状
        ffn_num_input = config.ffn_num_input
        # 前馈神经网络层的输入维度
        ffn_num_hiddens = config.ffn_num_hiddens
        # 前馈神经网络的隐藏层维度
        num_heads = config.num_heads
        # 多头注意力机制中头的数量
        num_layers = config.num_layers
        # Transformer中堆叠的层数
        dropout = config.dropout
        out_dim = config.out_dim
        # 分类器最后输出的维度
        news_size = config.news_size
        entity_size = config.entity_size
        topic_size = config.topic_size
        # 新闻节点、实体节点、话题节点的数量
        self.walk_length = config.walk_length
        self.output_size = config.d_model
        # 随机游走路径长度


        self.Transformer = TransformerEncoder(query_size, key_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, self.walk_length, dropout)
        # 一个Transformer编码器，负责对输入序列进行编码
        # TransformerEncoder模块将输入数据进行多层Transformer编码，输出隐藏表示
        self.decoder = nn.Linear(num_hiddens, news_size)
        # 定义一个全连接层，这层将Transformer的输出映射到新闻节点的空间，预测新闻相关信息
        self.clf = FNclf(num_hiddens, out_dim, dropout)
        # 定义一个分类器模块FNclf，

    def forward(self, X, device, args):
        h = self.Transformer(X, self.walk_length, args)
        # 将x传入Transformer编码器进行编码，Transformer根据输入得到一个隐藏表示h
        news_pred = 0
        scores, h_emb = self.clf(h)
        # scores：各个类别的得分(或概率)
        # h_emb:中间隐藏表示，用于进一步分析
        if args.case3 == "h0":
            scores = scores[:, 0, :]
        elif args.case3 == "mean":
            scores = scores.mean(dim=1)
        elif args.case3 == "max":
            scores = scores.max(dim=1)[0]
        # 根据args.case3的不同值，选择不同策略对scores进行聚合
        # h0:选择scores的第一项
        # mean：在维度1求平均
        # max：在维度1取最大值

        preds = torch.argmax(F.softmax(scores), dim=-1)
        # 对处理后的scores应用softmax归一化，得到概率分布，
        # 然后用torch.argmax找出每个样本中概率最大的类别索引作为预测值preds

        # return scores, preds, news_pred, h_emb[:, 0, :]
        return h_emb[:, 0, :]
        # scores:最终分类得分或者概率
        # preds： 预测的类别标签
        # news_pred:
        # h_emb[:,0,:]:从隐藏表示h_emb 选择第一个位置向量作为最终输出


'''  
def train(model, train_data, train_label, test_data, test_label, epochs, optimizer, device, args):
    epochs = tqdm(range(epochs))
    # tqdm可以看到训练的轮数(epochs),并在每个周期完成时更新进度
    train_loss_all = []
    # 存储每个训练周期的训练损失
    test_loss_all = []
    # 测试损失
    criteria_sup = torch.nn.CrossEntropyLoss()
    # 定义损失函数，这里使用的是CrossEntropyLoss，适用于多分类问题
    model = model.to(torch.float32)
    # 转换标签类型为 long
    train_label = train_label.to(device, dtype=torch.long)
    test_label = test_label.to(device, dtype=torch.long)

    train_X = genX(train_data, device)
    test_X = genX(test_data, device)
    # 使用genX将train_data和test_data转换为模型可接受的输入格式
    print(train_label.dtype)  # 应输出 torch.int64
    print(test_label.dtype)  # 应输出 torch.int64
    print(train_X.dtype)  # 应输出 torch.float32
    print(test_X.dtype)  # 应输出 torch.float32
    # 确保结果保存目录存在
    save_path = f"./results/{args.model}/auc"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_f1 = 0
    for epoch in epochs:
        optimizer.zero_grad()  
        train_scores, train_preds, train_news_decode, h_train = model(train_X, device, args)
        loss = criteria_sup(train_scores, train_label)
        train_loss_all.append(loss.item())
        loss.backward()  
        optimizer.step()  
        epochs.set_description(f"Training Epoch: {epoch}, Loss : {loss.item()}")
        with torch.no_grad():
            test_scores, test_preds, test_news_decode,h_test = model(test_X, device, args)
            test_loss = criteria_sup(test_scores, test_label) 
            test_loss_all.append(test_loss.item())
            test_res,auc_test = test_once(test_preds, test_scores, test_label)
           
            if test_res["test_f1_macro"] > best_f1:
                best_f1 = test_res["test_f1_macro"]
                best_test = test_res
                torch.save(auc_test, f"./results/{args.model}/auc/{args.dataset}_r{args.round}_auc.pt")
    with torch.no_grad():
        train_res,auc_trian = test_once(train_preds, train_scores, train_label)
        print_results_once(train_res, "train")
    with torch.no_grad():
        test_res,auc_test = test_once(test_preds, test_scores, test_label)
        print_results_once(test_res, "test")
        print("Best test results on Macro-F1:")
        print_results_once(best_test, "test")
    save_results(args, best_test)
'''