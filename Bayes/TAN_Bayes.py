import numpy as np
import math
from itertools import combinations
from scipy.stats import norm

class TAN:
    def __init__(self, smoothing=1e-10):
        self.classes = None
        self.class_priors = None
        self.feature_stats = None
        self.tree_structure = None
        self.smoothing = smoothing

    """计算特征之间的互信息矩阵"""
    def _compute_mutual_information(self, X, y):
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))

        for i, j in combinations(range(n_features), 2):
            mi = self._calculate_mi(X[:, i], X[:, j], y)
            mi_matrix[i, j] = mi_matrix[j, i] = mi

        return mi_matrix

    """计算两个特征和类别之间的互信息"""
    def _calculate_mi(self, x1, x2, y):
        unique_y = np.unique(y)
        unique_x1 = np.unique(x1)
        unique_x2 = np.unique(x2)

        mi = 0
        for yi in unique_y:
            py = np.mean(y == yi)
            for x1i in unique_x1:
                for x2i in unique_x2:
                    px1_y = np.mean((x1 == x1i) & (y == yi)) + self.smoothing
                    px2_y = np.mean((x2 == x2i) & (y == yi)) + self.smoothing
                    px1x2_y = np.mean((x1 == x1i) & (x2 == x2i) & (y == yi)) + self.smoothing

                    mi += px1x2_y * math.log(px1x2_y / (px1_y * px2_y))

        return mi

    """使用Chow-Liu算法最大化互信息来找到联合概率分布的最大生成树"""
    def _build_chow_liu_tree(self, mi_matrix):
        n_features = mi_matrix.shape[0]
        edges = []
        used_vertices = set()
        candidates = list(range(n_features))

        # 选择互信息最大的边
        while candidates:
            max_mi = -1
            best_edge = None

            for i, j in combinations(candidates, 2):
                if mi_matrix[i, j] > max_mi:
                    max_mi = mi_matrix[i, j]
                    best_edge = (i, j)

            if best_edge is None:
                break

            i, j = best_edge
            edges.append(best_edge)
            used_vertices.update([i, j])
            candidates = [v for v in candidates if v not in used_vertices]

        return edges

    """训练TAN模型"""
    def fit(self, X, y):
        # 计算先验概率
        self.classes = np.unique(y)
        self.class_priors = {c: (np.mean(y == c) + self.smoothing) for c in self.classes}

        # 计算特征互信息并构建树结构
        mi_matrix = self._compute_mutual_information(X, y)
        self.tree_structure = self._build_chow_liu_tree(mi_matrix)

        # 计算条件概率
        self.feature_stats = {}
        for c in self.classes:
            class_data = X[y == c]
            class_stats = {}

            # 离散和连续特征概率
            for i in range(X.shape[1]):
                unique_vals = np.unique(X[:, i])
                # 添加平滑处理
                class_stats[i] = {val: (np.mean(class_data[:, i] == val) + self.smoothing)
                                  for val in unique_vals}

            self.feature_stats[c] = class_stats

        return self

    """预测样本属于每个类别的概率"""
    def predict_proba(self, X):
        probas = []
        for x in X:
            class_probas = {}
            for c in self.classes:
                # 先验概率（已添加平滑）
                prob = math.log(self.class_priors[c])

                # 特征条件概率
                for i in range(len(x)):
                    # 使用平滑后的概率，确保不会出现对数域错误
                    discrete_prob = self.feature_stats[c][i].get(x[i], self.smoothing)
                    prob += math.log(discrete_prob)

                class_probas[c] = prob

            # 转换为概率
            max_log_prob = max(class_probas.values())
            exp_probs = {c: math.exp(p - max_log_prob) for c, p in class_probas.items()}
            total = sum(exp_probs.values())
            probas.append({c: p / total for c, p in exp_probs.items()})

        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return [max(proba, key=proba.get) for proba in probas]



raw_data = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
])

X = raw_data[:, :-1]  # 特征
y = raw_data[:, -1]  # 标签

# One-hot编码离散特征
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_encoded = X.copy()
for i in range(X.shape[1] - 2):
    X_encoded[:, i] = le.fit_transform(X[:, i])

X_encoded = X_encoded.astype(float)
y_encoded = (y == '好瓜').astype(int)  # 转换为二分类

# 初始化和训练模型
tan_model = TAN()
tan_model.fit(X_encoded, y_encoded)


# 在训练阶段保存编码器
encoders = []
for i in range(X.shape[1] - 2):
    le = LabelEncoder()
    le.fit(X[:, i])
    encoders.append(le)

test_samples = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
    ['浅白', '蜷缩', '沉闷', '模糊', '稍凹', '软粘', 0.463, 0.135],
    ['乌黑', '稍蜷', '清脆', '模糊', '凹陷', '硬滑', 0.428, 0.208],
])

test_encoded = test_samples.copy()
for i in range(test_samples.shape[1] - 2):
    test_encoded[:, i] = encoders[i].transform(test_samples[:, i])
test_encoded = test_encoded.astype(float)

# 预测
test_predictions = tan_model.predict(test_encoded)
test_probas = tan_model.predict_proba(test_encoded)

# 将数值标签转换回原始标签
test_predictions_label = ['好瓜' if p == 1 else '坏瓜' for p in test_predictions]

print("测试集预测结果:")
for i, (pred, proba) in enumerate(zip(test_predictions_label, test_probas), 1):
    print(f"样本 {i}: 预测为 {pred}")