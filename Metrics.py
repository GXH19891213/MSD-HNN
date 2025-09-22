
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


"""
	Utility functions for evaluating the model performance
"""

class Metrics(object):

	def __init__(self):
		super().__init__()

	def compute_metric(self, y_prob, y_true):
		k_list = ['Acc', 'F1', 'Pre', 'Recall']
		y_pre = np.array(y_prob).argmax(axis=1)
		# .argmax(axis=1) 返回沿着指定轴axis最大值的索引
		# 获取每个样本预测的类别标签
		size = len(y_prob)
		assert len(y_prob) == len(y_true)
		# 确保预测结果和真实标签数量一致

		scores = {str(k): 0.0 for k in k_list}
		scores['Acc'] += accuracy_score(y_true, y_pre) * size
		# 计算模型预测结果y_pre和真实标签y_true之间的准确率(即正确分类的样本比例) 加权
		scores['F1'] += f1_score(y_true, y_pre, average='macro') * size
		# average='macro'表示使用宏平均方法
		# macro平均是计算每个类别的F1分数后取平均，不考虑样本数量
		scores['Pre'] += precision_score(y_true, y_pre, zero_division=0) * size
		scores['Recall'] += recall_score(y_true, y_pre, zero_division=0) * size
		# zero_division=0是为了避免分母为0时的错误，类似于精确率的处理

		# y_true = np.array(y_true)
		# prob_log = y_prob[:, 1].tolist()
		#scores['auc'] = roc_auc_score(y_true, prob_log)

		return scores


