import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.models import HGNN
from torch.nn.parameter import Parameter
import dhg  # 深度超图库
import torch.nn.init as init
# 导入torch.nn.init模块，将其命名为init
from Layer1 import *
from TransformerBlock import TransformerBlock
import Constants
from dataLoader import *


class Fusion(nn.Module):
    #两级超图的融合层

    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        
        # hidden:这个子超图HGAT的输入，dy_emb: 这个子超图HGAT的输出
        # hidden和dy_emb都是用户embedding矩阵，大小(用户数,64)
        
        emb = torch.cat((hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)), dim=0)
        # hidden.unsqueeze(dim=0)在第0维增加一个维度，形状从(N,64)变为(1,N,64)
        # 最后沿着第0维拼接，形状为(2,,N,64)
        emb_score = nn.functional.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        # softmax输入形状是(2, N, 1)，输出 emb_score 的形状仍是 (2, N, 1)，但每个 (N, 1) 位置的两个值（对应 hidden 和 dy_emb）满足 softmax 归一化。
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        # 返回融合后的用户嵌入矩阵 (N, 64)
        return out



class Gated_fusion(nn.Module):
    # 最后融合两个特征向量

    def __init__(self, input_size, out_size=1, dropout=0.2):
        super(Gated_fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        # 使用Xavier 正态分布初始化权重，有助于保持信号在网络中传播的稳定性。
        init.xavier_normal_(self.linear2.weight)

    def forward(self, X1, X2):
        emb = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        # Dropout概率,由dropout参数设定，如0.2表示20%的神经元被丢弃，测试时会关闭Dropout
        out = torch.sum(emb_score * emb, dim=0)
        # 使用emb_score对emb加权求和，得到最终输出。
        # emb是(2,input_size)而emb_score是(2,1),逐元素相乘后求和，得到一个(input_size,)的向量
        return out


class HGSL(nn.Module):
    def __init__(self, opt):
        super(HGSL, self).__init__()
        self.hidden_size = opt.d_model
        self.n_node = opt.user_size
        # 定义隐藏维度和用户总数
        self.dropout = nn.Dropout(opt.dropout)
        self.initial_feature = opt.initialFeatureSize

        self.dynamic_hgnn = DynamicCasHGNN(self.initial_feature, self.hidden_size, dropout=opt.dropout)
        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature)
        # 初始化用户嵌入层，生成每个用户的初始特征向量
        self.stru_attention = TransformerBlock(self.hidden_size, n_heads=8)
        self.temp_attention = TransformerBlock(self.hidden_size, n_heads=8)
        # 初始化两个Transformer注意力模块 结构特征和时序特征
        self.global_cen_embedding = nn.Embedding(600, self.hidden_size)
        # 初始化全局中心嵌入层
        self.cas_pos_embedding = nn.Embedding(50, self.hidden_size)
        # 初始化传播位置嵌入层
        self.local_inf_embedding = nn.Embedding(200, self.hidden_size)
        # 初始化用户行为嵌入层 编码用户影响力user_inf
        self.weight = Parameter(torch.Tensor(self.hidden_size + 2, self.hidden_size + 2))
        self.weight2 = Parameter(torch.Tensor(self.hidden_size + 2, self.hidden_size + 2))
        # 初始化两个可学习权重矩阵，用于时序和结构表示的线性变换
        self.fus = Gated_fusion(self.hidden_size + 2)
        # 初始化门控融合模块，融合时序和结构表示
        self.linear = nn.Linear((self.hidden_size + 2), 32)
        # 32改成了64 3.31
        self.reset_parameters()
        # 参数从均匀分布初始化

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data_idx, hypergraph_list, data_name=None):
        news_centered_graph, user_centered_graph, spread_status = GraphReader(data_name)
        seq, timestamps, user_level = (item for item in news_centered_graph)
        # 用户序列 时间戳 用户等级
        useq, user_inf, user_cen = (item for item in user_centered_graph)
        # 新闻序列 用户影响力 用户活跃度

        # Global learning
        # hidden = self.dropout(self.user_embedding.weight)
        hidden = self.user_embedding.weight
        # user_cen = self.global_cen_embedding(user_cen)
        # tweet_hidden = hidden + user_cen
        tweet_hidden = hidden
        # tweet_hidden形状是torch.Size([28049, 64])
        print("tweet_hidden 形状:", tweet_hidden.shape)
        # 动态超图卷积
        user_hgnn_out = self.dynamic_hgnn(tweet_hidden, hypergraph_list)
        # 生成全局用户嵌入

        # Normalize
        zero_vec1 = -9e15 * torch.ones_like(seq[data_idx])
        one_vec = torch.ones_like(seq[data_idx], dtype=torch.float)
        nor_input = torch.where(seq[data_idx] > 0, one_vec, zero_vec1)
        nor_input = F.softmax(nor_input, 1)
        att_mask = (seq[data_idx] == Constants.PAD)
        adj_with_fea = F.embedding(seq[data_idx], user_hgnn_out)


        # 移除 global_time = self.local_time_embedding(timestamps[data_idx])
        # 直接使用 adj_with_fea，不再添加时间嵌入
        att_hidden = adj_with_fea
        att_out = self.temp_attention(att_hidden, att_hidden, att_hidden, mask=att_mask)
        news_out = torch.einsum("abc,ab->ac", (att_out, nor_input))
        # 时序注意力机制
        # Concatenate temporal propagation status
        news_out = torch.cat([news_out, spread_status[data_idx][:, 2:] / 3600 / 24], dim=-1)
        # 拼接传播状态时间特征(T1,T2)
        news_out = news_out.matmul(self.weight)
        # 线性变换 调整news_out维度


        local_inf = self.local_inf_embedding(user_inf[data_idx])
        # 用户影响力嵌入  用户活跃行为（如转发强度）
        cas_pos = self.cas_pos_embedding(user_level[data_idx])
        # 传播位置嵌入 用户等级
        att_hidden_str = adj_with_fea + local_inf + cas_pos
        # 融合全局嵌入 行为嵌入和位置嵌入
        att_out_str = self.stru_attention(att_hidden_str, att_hidden_str, att_hidden_str, mask=att_mask, pos=False)
        news_out_str = torch.einsum("abc,ab->ac", (att_out_str, nor_input))
        # 加权聚合生成结构表示

        news_out_str = torch.cat([news_out_str, spread_status[data_idx][:, :2]], dim=-1)
        # 拼接传播状态的结构特征 S1子级联数量,S2非孤立级联比例
        news_out_str = news_out_str.matmul(self.weight2)
        # 线性变换
        print(f"news_out mean: {news_out.mean()}, std: {news_out.std()}")
        print(f"news_out_str mean: {news_out_str.mean()}, std: {news_out_str.std()}")
        # Gated fusion
        news_out = self.fus(news_out, news_out_str)
        output = self.linear(news_out)

        return output

class DynamicCasHGNN(nn.Module):
    '''超图HGNN'''

    def __init__(self, input_num, embed_dim, step_split=8, dropout=0.5, is_norm=False):
        super().__init__()
        self.input_num = input_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.is_norm = is_norm
        self.step_split = step_split
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(self.embed_dim)
        self.user_embeddings = nn.Embedding(self.input_num, self.embed_dim)
        self.hgnn = HypergraphConv(self.embed_dim, self.embed_dim, drop_rate=self.dropout)
        self.fus = Fusion(self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        '''从正态分布中随机初始化每张超图的初始用户embedding'''
        init.xavier_normal_(self.user_embeddings.weight)

    def forward(self, tweet_hidden, hypergraph_list, device=torch.device('cuda')):
        # tweet_hidden 是用户的嵌入特征
        # hypergraph_list 是超图序列

        hg_embeddings = []
        for i in range(len(hypergraph_list)):
            # 对超图序列中每个超图进行卷积处理，每次通过self.hgnn将当前超图用户嵌入进行卷积
            subhg_embedding = self.hgnn(tweet_hidden, hypergraph_list[i])  # 使用 tweet_hidden 作为用户特征
            if i == 0:
                hg_embeddings.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
                hg_embeddings.append(subhg_embedding)
        # 返回最后一个时刻的用户embedding
        return hg_embeddings[-1]
class HGNN_Model(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5, user_size=1000, device='cpu'):
        super(HGNN_Model, self).__init__()

        # 用户嵌入层
        self.user_embedding = nn.Embedding(user_size, input_size)

        # 动态超图卷积模块
        self.dynamic_hgnn = DynamicCasHGNN(input_size, output_size, dropout=dropout)

        # 其他模块（如分类器等）
        self.classifier = nn.Linear(output_size, 2)  # 假设是二分类

    def forward(self, user_ids, hypergraph_list):
        # 获取用户的嵌入特征
        X = self.user_embedding(user_ids)

        # 获取动态超图卷积层的输出
        X = self.dynamic_hgnn(X, hypergraph_list)

        # 分类
        output = self.classifier(X)
        output = F.log_softmax(output, dim=1)

        return output


