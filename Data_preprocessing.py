import itertools
import numpy as np
import torch
import Constants
import pickle
import os
import json

# 创建必要的目录
processed_dir = 'data/poli/Processed'
os.makedirs(processed_dir, exist_ok=True)
processed_dir1 = ('data/gos/Processed')
os.makedirs(processed_dir1, exist_ok=True)


class Options(object):
    def __init__(self, data_name='poli'):
        self.nretweet = 'data/' + data_name + '/news_centered_data.txt'
        # 存储新闻中心化数据文件路径，新闻中心化数据以新闻为中心组织的数据，可能包含每条新闻及相关信息
        self.uretweet = 'data/' + data_name + '/user_centered_data.txt'
        self.label = 'data/' + data_name + '/label.txt'
        # 存储新闻标签
        self.news_list = 'data/' + data_name + '/' + data_name + '_news_list.txt'
        # 存储新闻列表
        self.news_centered = 'data/' + data_name + '/Processed/news_centered.pickle'
        self.user_centered = 'data/' + data_name + '/Processed/user_centered.pickle'
        self.train_idx = torch.from_numpy(np.load('data/' + data_name + '/train_idx.npy'))
        self.valid_idx = torch.from_numpy(np.load('data/' + data_name + '/val_idx.npy'))
        self.test_idx = torch.from_numpy(np.load('data/' + data_name + '/test_idx.npy'))
        # 这些变量存储训练集、验证集和测试集索引。这些索引用于划分数据集，指示哪些数据用于训练模型，哪些用于验证模型性能，哪些用于测试模型的最终性能。
        # torch.from_numpy将numpy数组转换成等价的pytoch张量
        self.train = 'data/' + data_name + '/Processed/train_processed.pickle'
        self.valid = 'data/' + data_name + '/Processed/valid_processed.pickle'
        self.test = 'data/' + data_name + '/Processed/test_processed.pickle'
        self.user_mapping = 'data/' + data_name + '/user_mapping.pickle'
        # 存储用户映射文件的路径，user_mapping.pickle文件通常包含了用户用户ID到连续整数索引的映射关系。
        # 在处理大规模数据集时，经常需要将用户ID（可能是字符串或非连续的数字）转换为连续的整数索引
        self.news_mapping = 'data/' + data_name + '/news_mapping.pickle'
        self.save_path = ''
        # 用于保存训练好的模型和中间结果
        self.embed_dim = 64
        #  指定嵌入向量的维度


def buildIndex(user_set, news_set):
    n2idx = {}
    # 用于存储新闻到索引的映射的字典
    u2idx = {}
    # 存储用户到索引的映射
    pos = 0
    # 记录当前索引值
    # 更新pos，为下个用户分配索引
    for user in user_set:
        # 开始一个循环，遍历user_set中的每个用户
        u2idx[user] = pos
        # 对于每个用户，将用户作为键，当前值的pos值作为索引，存储到u2idx字典
        pos += 1
    pos = 0
    for news in news_set:
        n2idx[news] = pos
        pos += 1
    user_size = len(user_set)
    news_size = len(news_set)
    return user_size, news_size, u2idx, n2idx


# 处理传播链数据的函数，不使用填充零值
def Pre_data(data_name, idx, label, max_len=600):
    options = Options(data_name)
    cascades = {}

    ''' load news-centered retweet data'''
    for line in open(options.nretweet):
        userlist = []
        timestamps = []
        levels = []
        infs = []

        chunks = line.strip().split(',')
        cascades[chunks[0]] = []
        # 以新闻ID为键，初始化新闻的传播链数据
        for chunk in chunks[1:]:
            try:
                user, timestamp, level, inf = chunk.split()
                userlist.append(user)
                timestamps.append(float(timestamp) / 3600 / 24)  # 转换为天数
                levels.append(int(level) + 1)  # 层级从0开始
                infs.append(inf)
            except:
                user = chunk
                userlist.append(user)
                timestamps.append(None)  # 标记为 None，表示无效时间戳
                infs.append(None)  # 无效影响力
                levels.append(None)  # 无效层级
                print('tweet root', chunk)
                # 处理的是根用户或有问题的数据
        cascades[chunks[0]] = [userlist, timestamps, levels, infs]
        # 以新闻ID为键，初始化新闻的传播链数据

    batch_news_list = idx.tolist()  # 获取当前批次中的新闻ID列表
    # print(batch_news_list)
    for idx, cas in enumerate(cascades.keys()):
        max_ = max_len
        # 截取传播链数据前max_元素
        cascades[cas] = [i[:max_] for i in cascades[cas]]
        # 根据时间戳排序
        order = [i[0] for i in sorted(enumerate(cascades[cas][1]), key=lambda x: float(x[1]))]
        cascades[cas] = [[x[i] for i in order] for x in cascades[cas]]

    '''load user-centered retweet data'''
    ucascades = {}
    for line in open(options.uretweet):
        newslist = []
        userinf = []
        chunks = line.strip().split(',')
        ucascades[chunks[0]] = []

        for chunk in chunks[1:]:
            news, timestamp, inf = chunk.split()
            newslist.append(news)
            userinf.append(inf)

        ucascades[chunks[0]] = np.array([newslist, userinf])

    '''ordered by userinf'''
    for cas in list(ucascades.keys()):
        order = [i[0] for i in sorted(enumerate(ucascades[cas][1]), key=lambda x: float(x[1]))]
        ucascades[cas] = [[x[i] for i in order] for x in ucascades[cas]]
    user_set = ucascades.keys()
    # 键：用户ID 值：newslist, userinf两个列表

    ''' 检查用户索引文件是否存在，存在从中加载用户索引，不存在建立文件 '''
    if os.path.exists(options.user_mapping):
        with open(options.user_mapping, 'rb') as handle:
            u2idx = pickle.load(handle)
            user_size = len(list(user_set))
            # print(f"user_size: {user_size}")
        with open(options.news_mapping, 'rb') as handle:
            n2idx = pickle.load(handle)
            # news_size = len(batch_news_list)
            # print(f"news_size: {news_size}")
    else:
        user_size, news_size, u2idx, n2idx = buildIndex(user_set, news_list)
        with open(options.user_mapping, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.news_mapping, 'wb') as handle:
            pickle.dump(n2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # HIGHEST_PROTOCOL，当前python版本支持的最高协议

    '''替换用户和新闻索引，按批次索引保存传播链'''
    for cas in cascades:
        cascades[cas][0] = [u2idx[u] for u in cascades[cas][0]]
        # 将cascades[cas][0]中每个用户u替换成其对应索引
    t_cascades = dict([(n2idx[key], cascades[key]) for key in cascades])
    # print(t_cascades)
    # 构建字典，键是新闻索引，值是原始cascades 中对应的传播链数据
    cascades1 = {key: value for key, value in t_cascades.items() if key in batch_news_list}
    cascades = {key: cascades1[key] for key in batch_news_list if key in cascades1}
    for cas in ucascades:
        ucascades[cas][0] = [n2idx[n] for n in ucascades[cas][0]]
        # 将ucascades[cas][0]中每个新闻替换成索引
    u_cascades = dict([(u2idx[key], ucascades[key]) for key in ucascades])
    # 构建字典，键是用户索引，值是原始ucascades中对应的传播链
    # news_list = [0] + news_list
    # 创建存储级联数据的字典
    # 使用字典来存储每个新闻对应的seq, timestamps, user_level, user_inf
    seq = {}
    timestamps = {}
    user_level = {}
    user_inf = {}

    for n, s in cascades.items():
        seq[n] = s[0]  # 将第一个列表保存到seq字典
        timestamps[n] = s[1]  # 将第二个列表保存到timestamps字典
        user_level[n] = s[2]  # 将第三个列表保存到user_level字典
        user_inf[n] = s[3]  # 将第四个列表保存到user_inf字典
        # print(seq.keys())
        # seq键是新闻列表 值是用户序列
    # 将字典转换为列表
    seq_list = list(seq.values())
    timestamps_list = list(timestamps.values())
    user_level_list = list(user_level.values())
    user_inf_list = list(user_inf.values())

    def convert_to_float64(lst):
        result = []
        for sublist in lst:
            new_sublist = [np.float64(item) for item in sublist]
            result.append(new_sublist)
        return result

    seq = convert_to_float64(seq_list)
    timestamps = convert_to_float64(timestamps_list)
    user_level = convert_to_float64(user_level_list)
    user_inf = convert_to_float64(user_inf_list)

    # 将转换后的列表组织成news_cascades
    news_cascades = [seq, timestamps, user_level, user_inf]

    # print(news_cascades)
    return news_cascades, user_size


def data1(data_name, early_type, early, max_len=600):
    options = Options(data_name)
    cascades = {}

    '''load news-centered retweet data'''
    for line in open(options.nretweet):
        userlist = []
        timestamps = []
        levels = []
        infs = []
        # politifact10903,5947098064 142802990.0 0 4,3151879206 142828443.0 1 1
        chunks = line.strip().split(',')
        cascades[chunks[0]] = []
        for chunk in chunks[1:]:
            # 5947098064 142802990.0 0 4
            try:
                user, timestamp, level, inf = chunk.split()
                userlist.append(user)
                timestamps.append(float(timestamp) / 3600 / 24)
                levels.append(int(level) + 1)
                # 原值从0开始
                infs.append(inf)

            except:
                user = chunk
                userlist.append(user)
                timestamps.append(float(0.0))
                infs.append(1)
                levels.append(1)
                print('tweet root', chunk)
                # 提示处理的是根用户或者有问题的数据
        cascades[chunks[0]] = [userlist, timestamps, levels, infs]


    news_list = []
    for line in open(options.news_list):
        news_list.append(line.strip())
    cascades = {key: value for key, value in cascades.items() if key in news_list}
    '''根据不同条件截断传播链数据'''

    '''ordered by timestamps'''
    for idx, cas in enumerate(cascades.keys()):
        # 遍历字典的键，返回每个键的索引idx和对应的键cas
        max_ = max_len
        cascades[cas] = [i[:max_] for i in cascades[cas]]
        # 截取传播链数据的前max_元素
        order = [i[0] for i in sorted(enumerate(cascades[cas][1]), key=lambda x: float(x[1]))]
        cascades[cas] = [[x[i] for i in order] for x in cascades[cas]]
    ucascades = {}
    '''load user-centered retweet data'''
    for line in open(options.uretweet):
        # 5835043030,politifact15545 343379794.0 1
        newslist = []
        userinf = []

        chunks = line.strip().split(',')

        ucascades[chunks[0]] = []

        for chunk in chunks[1:]:
            news, timestamp, inf = chunk.split()
            newslist.append(news)
            userinf.append(inf)
        ucascades[chunks[0]] = np.array([newslist, userinf])

    '''ordered by userinf'''
    for cas in list(ucascades.keys()):
        order = [i[0] for i in sorted(enumerate(ucascades[cas][1]), key=lambda x: float(x[1]))]
        ucascades[cas] = [[x[i] for i in order] for x in ucascades[cas]]
    user_set = ucascades.keys()
    # 键：用户ID 值：newslist, userinf两个列表
    if os.path.exists(options.user_mapping):
        with open(options.user_mapping, 'rb') as handle:
            u2idx = pickle.load(handle)
            user_size = len(list(user_set))
        with open(options.news_mapping, 'rb') as handle:
            n2idx = pickle.load(handle)
            news_size = len(news_list)
            print(news_size)
    else:
        user_size, news_size, u2idx, n2idx = buildIndex(user_set, news_list)
        with open(options.user_mapping, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.news_mapping, 'wb') as handle:
            pickle.dump(n2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # HIGHEST_PROTOCOL，当前python版本支持的最高协议
    '''替换索引 得到 t_cascades u_cascades'''
    for cas in cascades:
        cascades[cas][0] = [u2idx[u] for u in cascades[cas][0]]
        # 将cascades[cas][0]中每个用户u替换成其对应索引
    t_cascades = dict([(n2idx[key], cascades[key]) for key in cascades])
    # 构建字典，键是新闻索引，值是原始cascades 中对应的传播链数据
    for cas in ucascades:
        ucascades[cas][0] = [n2idx[n] for n in ucascades[cas][0]]
        # 将cascades[cas][0]中每个用户新闻替换成索引
    u_cascades = dict([(u2idx[key], ucascades[key]) for key in ucascades])
    # 构建字典，键是用户索引，值是原始ucascades中对应的传播链

    '''load labels'''
    labels = np.zeros((news_size, 1))
    # news_size行、1列
    for line in open(options.label):
        news, label = line.strip().split(' ')
        if news in n2idx:
            labels[n2idx[news]] = label
    seq = np.zeros((news_size, max_len))
    timestamps = np.zeros((news_size, max_len))
    user_level = np.zeros((news_size, max_len))
    user_inf = np.zeros((news_size, max_len))
    # 将新闻ID存储到news_list列表的正确位置
    for n, s in cascades.items():
        news_list[n2idx[n]] = n
        se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
        seq[n2idx[n]] = se_data

        t_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))
        timestamps[n2idx[n]] = t_data

        lv_data = np.hstack((s[2], np.array([Constants.PAD] * (max_len - len(s[2])))))
        user_level[n2idx[n]] = lv_data

        inf_data = np.hstack((s[3], np.array([Constants.PAD] * (max_len - len(s[3])))))
        user_inf[n2idx[n]] = inf_data
        # print(f"user_inf:{user_inf[n2idx[n]]}")
        # print(seq.keys())
        # seq键是新闻列表 值是用户序列
        # 将字典转换为列表
    useq = np.zeros((user_size, max_len))
    uinfs = np.zeros((user_size, max_len))

    for n, s in ucascades.items():
        if len(s[0]) < max_len:
            se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
            useq[u2idx[n]] = se_data

            tinf_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))
            uinfs[u2idx[n]] = tinf_data
        else:
            useq[u2idx[n]] = s[0][:max_len]
            # utimestamps[u2idx[n]] = s[1][:max_len]
            uinfs[u2idx[n]] = s[1][:max_len]
    total_len = sum(len(t_cascades[i][0]) for i in t_cascades)
    # 计算所有新闻的传播链总长度（总的传播用户数）
    # t_cascades[i]是一个包含传播链数据的列表，t_cascades[i][0]访问该新闻的传播序列（即新闻传播的用户ID列表）
    total_ulen = sum(len(u_cascades[i][0]) for i in u_cascades)
    # u_cascades是记录了每个用户的参与情况，u_cascades[i] 对应某个用户的传播链数据，
    # u_cascades[i][0] 访问该用户的参与序列（即用户参与的新闻列表）。
    print("total size:%d " % (len(seq)))
    # 打印新闻传播序列的总数量）
    print('spread size', (total_len))
    # 打印所有新闻传播链的总长度（即所有新闻传播的总用户数）
    print("average news cascades length:%f" % (total_len / (len(seq))))
    print("average user participant length:%f" % (total_ulen / (len(useq))))
    print("user size:%d" % (user_size))
    news_cascades = [seq, timestamps, user_level, user_inf]
    user_parti = [useq, uinfs]
    return news_cascades, user_parti, labels, user_size, news_list


if __name__ == "__main__":
    data_name = ('poli')
    options = Options(data_name)
    news_cascades, user_parti, labels, user_size, news_list = data1(data_name, early_type=Constants.early_type,
                                                                    early=None)
    train_news = np.array([i for i in options.train_idx])
    valid_news = np.array([i for i in options.valid_idx])
    test_news = np.array([i for i in options.test_idx])

    train_data = [train_news, labels[train_news]]
    valid_data = [valid_news, labels[valid_news]]
    test_data = [test_news, labels[test_news]]

    with open(options.news_centered, 'wb') as handle:
        pickle.dump(news_cascades, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.user_centered, 'wb') as handle:
        pickle.dump(user_parti, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(options.train, 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.valid, 'wb') as handle:
        pickle.dump(valid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.test, 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
