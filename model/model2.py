import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, to_hetero
from torch_geometric.data import Data
from torch.nn import Linear, ReLU
from torch_geometric.transforms import ToUndirected

class Model2(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super(Model2, self).__init__(**kwargs)
        # **kwargs 可选参数
        # Set device for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Store the configuration for later use
        self.config = config
        # 添加output_size属性
        self.output_size = 64
        # self.dropout = nn.Dropout(0.2)
        # 添加正则化的部分
        # Define the GNN layers for each node type and edge type
        self.hetero_conv = HeteroConv({
            ('news', 'has', 'entity'): SAGEConv((-1, -1), 64),  # SAGEConv for 'news' nodes
            ('news', 'belongs', 'topic'): GATConv((-1, -1), 64, add_self_loops=False),  # GATConv for 'entity' nodes
            ('news', 'links', 'news'): GATConv((-1, -1), 64, add_self_loops=False)
        }, aggr='sum')

        # Define the linear layers and activation functions
        self.linear1 = Linear(64, 64)
        # 上面的128 之前是64 以及下面的128   64是32变的 3.31
        self.relu1 = ReLU(inplace=True)
        # 创建激活函数RelU，inplace=True表示直接在输入张量上进行修改，以节省内存
        self.linear2 = Linear(64, 32)

    def forward(self, data, news_indices):
        data = data.to(self.device)

        # Transform the data if needed (e.g., ToUndirected)
        # transform = ToUndirected(merge=True)
        # data = transform(data)
        # 将图数据转换为无向图
        # Get the embeddings for each node type using HeteroGraphConv
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        # 从图数据中提取节点特征字典和边索引字典
        x_dict = self.hetero_conv(x_dict, edge_index_dict)
        # 通过HeteroConv计算每个节点类型的嵌入
        # Apply linear layer and ReLU activation to the 'news' nodes
        x_news = self.linear1(x_dict['news'])
        x_news = self.relu1(x_news)

        # Final linear layer for output
        out = self.linear2(x_news)
        # print("out shape:", out.shape)
        '''
        relevant_out = out[news_indices]
        # print("relevant_out:", relevant_out)
        # Return the 'news' node output
        return relevant_out
        '''
        # 获取 n_id（子图中新闻节点的全局索引）
        n_id = data['news'].n_id  # [num_news_nodes], 例如 [41, 288, ..., 272]
        # print(f"n_id: {n_id}")
        # print(f"news_indices: {news_indices}")

        # 将全局索引转换为局部索引
        local_indices = torch.zeros_like(news_indices, dtype=torch.long)
        for i, global_idx in enumerate(news_indices):
            local_idx = (n_id == global_idx).nonzero(as_tuple=True)[0]
            if local_idx.numel() > 0:
                local_indices[i] = local_idx[0]
            else:
                raise ValueError(f"Global index {global_idx} not found in n_id: {n_id}")
        # print(f"local_indices: {local_indices}")

        # 提取对应节点的输出
        relevant_out =out[local_indices]  # [batch_size, 32], 例如 [32, 32]
        # print("relevant_out shape:", relevant_out.shape)

        return relevant_out
