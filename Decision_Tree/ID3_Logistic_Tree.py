import pandas as pd
import numpy as np

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 对率回归训练函数（使用牛顿法）
def train(X, y, iterations=150, regularization=1e-6):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))  # 添加偏置项
    theta = np.zeros(n + 1)

    for _ in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)

        # 计算梯度
        gradient = np.dot(X.T, (h - y)) / m

        # 计算海森矩阵
        diag_h = h * (1 - h)
        H = np.dot(X.T, diag_h[:, None] * X) / m

        # 添加正则化项
        H += regularization * np.eye(H.shape[0])

        # 更新参数
        theta -= np.linalg.inv(H).dot(gradient)

    return theta

# 对率回归预测函数
def predict(X, theta):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加偏置项
    return sigmoid(np.dot(X, theta))

def info_entropy(data):  # 求信息熵函数
    data_label = data.iloc[:, -1]
    label_class = data_label.value_counts()
    Ent = 0
    for k in label_class.keys():
        p_k = label_class[k] / len(data_label)
        Ent += -p_k * np.log2(p_k)
    return Ent


def info_gain(data, a):  # 求离散属性的信息增益
    Ent = info_entropy(data)
    feature_class = data[a].value_counts()
    gain = 0
    for v in feature_class.keys():
        weight = feature_class[v] / data.shape[0]
        Ent_v = info_entropy(data.loc[data[a] == v])
        gain += weight * Ent_v
    return Ent - gain


def info_gain_continuous(data, a):  # 求连续属性的信息增益和最佳划分点
    n = len(data)
    data_sorted = data.sort_values(by=a).reset_index(drop=True)
    Ent = info_entropy(data)

    max_gain = 0
    best_split = None

    for i in range(n - 1):
        if data_sorted.iloc[i, data_sorted.columns.get_loc(a)] != data_sorted.iloc[
            i + 1, data_sorted.columns.get_loc(a)]:
            val = (data_sorted.iloc[i, data_sorted.columns.get_loc(a)] + data_sorted.iloc[
                i + 1, data_sorted.columns.get_loc(a)]) / 2
            data_left = data[data[a] <= val]
            data_right = data[data[a] > val]
            ent_left = info_entropy(data_left)
            ent_right = info_entropy(data_right)
            gain = Ent - (len(data_left) / n) * ent_left - (len(data_right) / n) * ent_right

            if gain > max_gain:
                max_gain = gain
                best_split = val

    return max_gain, best_split

# 基于对率回归寻找最佳划分点（用于连续特征）
def logistic_info_gain(data, feature):
    X = data[[feature]].values
    y = (data.iloc[:, -1] == data.iloc[:, -1].unique()[0]).astype(int).values  # 二分类

    theta = train(X, y)
    predictions = predict(X, theta) >= 0.5
    left_group = data[predictions]
    right_group = data[~predictions]

    # 计算分组后的信息增益
    Ent = info_entropy(data)
    gain = Ent - (len(left_group) / len(data)) * info_entropy(left_group) - (len(right_group) / len(data)) * info_entropy(right_group)

    return gain, theta

def get_best_fea(data):  # 找到最大信息增益的属性
    features = data.columns[:-1]
    res = {}
    best_split_points = {}

    for a in features:
        if data[a].dtype == 'object':  # 离散特征
            gain = info_gain(data, a)
        else:  # 连续特征
            gain, split = info_gain_continuous(data, a)
            best_split_points[a] = split  # 存储连续特征的最佳划分点

        res[a] = gain

    best_feature = max(res, key=res.get)

    # 最佳特征是连续特征，返回划分点
    return (best_feature, best_split_points[best_feature] if best_feature in best_split_points else None)


def get_most_label(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.index[0]


def drop_exist_feature(data, best_feature, split_value=None):#基于信息熵
    if split_value is None:  # 离散特征
        attr = pd.unique(data[best_feature])
        new_data = [(nd, data[data[best_feature] == nd].drop([best_feature], axis=1)) for nd in attr]
    else:  # 连续特征
        new_data = [('<= {}'.format(split_value), data[data[best_feature] <= split_value].drop([best_feature], axis=1)),
                    ('> {}'.format(split_value), data[data[best_feature] > split_value].drop([best_feature], axis=1))]
    return new_data


def drop_exist_feature_logistic(data, best_feature, theta=None):# 基于对率回归
    if theta is None:  # 离散特征
        attr = pd.unique(data[best_feature])
        new_data = [(v, data[data[best_feature] == v].drop([best_feature], axis=1)) for v in attr]
    else:  # 连续特征
        predictions = predict(data[[best_feature]].values, theta) >= 0.5
        new_data = [('<= 0.5', data[predictions].drop([best_feature], axis=1)),
                    ('> 0.5', data[~predictions].drop([best_feature], axis=1))]

    return new_data

def create_ID3_Tree(data):#基于信息熵的决策树
    data_label = data.iloc[:, -1]
    if len(data) == 0:
        return None
    if len(data_label.unique()) == 1:
        return data_label.iloc[0]
    if data.shape[1] == 1:
        return get_most_label(data)

    best_fea, split_value = get_best_fea(data)
    tree = {best_fea: {}}

    for attr_value, subset in drop_exist_feature(data, best_fea, split_value):
        tree[best_fea][attr_value] = create_ID3_Tree(subset)

    return tree

def create_logistic_tree(data):#基于对率回归的决策树
    data_label = data.iloc[:, -1]
    if len(data) == 0:
        return None
    if len(data_label.unique()) == 1:
        return data_label.iloc[0]
    if data.shape[1] == 1:
        return get_most_label(data)

    best_fea, theta = get_best_fea(data)
    tree = {best_fea: {}}

    for attr_value, subset in drop_exist_feature_logistic(data, best_fea, theta):
        tree[best_fea][attr_value] = create_logistic_tree(subset)

    return tree


def predict_tree(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    feature_value = sample.get(feature)

    if feature_value is None:
        return get_most_label(pd.DataFrame([sample]))

    for key in tree[feature]:
        if key.startswith('<=') and feature_value <= float(key.split()[-1]):
            return predict_tree(tree[feature][key], sample)
        elif key.startswith('>') and feature_value > float(key.split()[-1]):
            return predict_tree(tree[feature][key], sample)
        elif feature_value == key:
            return predict_tree(tree[feature][key], sample)

    return get_most_label(pd.DataFrame([sample]))


#画出决策树结构
def print_tree(tree, indent=''):
    if not isinstance(tree, dict):
        print(indent + str(tree))
        return

    for node, subtree in tree.items():
        print(indent + str(node))
        for edge, child in subtree.items():
            print(indent + '  ' + f'[{edge}] ->', end=' ')
            print_tree(child, indent + '    ')




# 主程序
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 读取数据
    file_path = 'D:/pycharm_codes/Watermelon_3.csv'
    data = pd.read_csv(file_path)

    # 分割数据为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 创建决策树
    decision_tree = create_ID3_Tree(train_data)
    logistic_tree = create_logistic_tree(data)


    print("基于信息熵的决策树结构:")
    print_tree(decision_tree)

    # 测试准确率
    correct_predictions = 0

    for index, row in test_data.iterrows():
        sample = row[:-1].to_dict()  # 提取特征列作为样本
        actual_label = row.iloc[-1]  # 提取实际标签
        predicted_label = predict_tree(decision_tree, sample)  # 预测结果

        if predicted_label == actual_label:
            correct_predictions += 1  # 计数正确预测

    accuracy = correct_predictions / len(test_data)  # 计算准确率
    print(f'基于信息熵的决策树准确率: {accuracy:.2%}')

    print('\n')

    print("基于对率回归的决策树结构:")
    print_tree(logistic_tree)
    correct_predictions = 0
    for index, row in test_data.iterrows():
        sample = row[:-1].to_dict()  # 提取特征列作为样本
        actual_label = row.iloc[-1]  # 提取实际标签
        predicted_label = predict_tree(logistic_tree, sample)  # 预测结果

        if predicted_label == actual_label:
            correct_predictions += 1  # 计数正确预测

    accuracy = correct_predictions / len(test_data)  # 计算准确率
    print(f'基于对率回归的决策树准确率: {accuracy:.2%}')
