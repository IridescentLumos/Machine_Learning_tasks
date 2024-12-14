import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns # 创建统计图形
import cvxopt

class LinearSVM:
    def __init__(self, C=None):
        """
        C:软硬间隔的参数，C=0表示硬间隔
        """
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 构造核矩阵
        K = np.dot(X, X.T)

        # 转为二次规划问题的参数
        P = cvxopt.matrix(np.outer(y, y) * K) #二次项
        q = cvxopt.matrix(-np.ones(n_samples)) #线性项
        A = cvxopt.matrix(y, (1, n_samples), tc='d') #对偶问题的约束
        b = cvxopt.matrix(0.0) #偏置项

        if self.C is None:  # 硬间隔
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:  # 软间隔
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # 求解二次规划问题
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x']) # 拉格朗日乘子

        # 支持向量
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # 权重向量
        self.w = np.dot(self.sv_y * self.alphas, self.sv)

        # 偏置项
        self.b = np.mean(self.sv_y - np.dot(self.sv, self.w))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

file_path = 'D:/pycharm_codes/MachineLearning/SVM/data.csv'
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
# 数据特征可视化
print(data.columns)

# diagnosis中B是benign良性，M是malignant恶性
# 数据特征中：_mean是均值，_se是标准差，_worst是最大值，因此划分特征为3段
features_mean=list(data.columns[2:12])
features_se=list(data.columns[12:22])
features_worst=list(data.columns[22:32])


# 进行数据预处理,去除id一列，改B为-1，M为1
data.drop('id',axis=1,inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': -1}).astype(int)


# 诊断结果可视化
#sns.countplot(data['diagnosis'],label='Count')
#plt.show()

# 热力图呈现_mean的特征之间的相关性
corr=data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()

# 因为特征较多，所以选取一部分特征用于进行SVM训练，不使用所有特征
# 选取_mean的特征，不使用_se,_worst
# 由上面的热力图可以分析出相关性较强的若干特征，可以将这些特征中选取一个作为代表，简化特征个数
features_remain=['perimeter_mean','concavity_mean','texture_mean','smoothness_mean','symmetry_mean', 'fractal_dimension_mean']

# 接下来进行数据的打散随机与6:2:2的训练集验证集测试集划分
X = data[features_remain]
y = data['diagnosis']

# 随机打乱数据
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算划分数量
n_total = len(data_shuffled)
n_train = int(0.6 * n_total)  # 60%
n_val =n_test=int(0.2 * n_total)    # 20%


# 数据分割
train_data = data_shuffled.iloc[:n_train]
val_data = data_shuffled.iloc[n_train:n_train + n_val]
test_data = data_shuffled.iloc[n_train + n_val:]

# 提取特征和标签
X_train, y_train = train_data[features_remain], train_data['diagnosis']
X_val, y_val = val_data[features_remain], val_data['diagnosis']
X_test, y_test = test_data[features_remain], test_data['diagnosis']

# 打印划分结果
print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")


from sklearn.preprocessing import StandardScaler

# 数据准备
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

# 对数据规范化,保证数据维度一致
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 训练线性核SVM（软间隔和硬间隔分别训练）
print("=== Hard Margin SVM ===")
hard_svm = LinearSVM(C=None)
hard_svm.fit(X_train, y_train)
hard_train_acc = np.mean(hard_svm.predict(X_train) == y_train)
hard_val_acc = np.mean(hard_svm.predict(X_val) == y_val)
print(f"Training Accuracy: {hard_train_acc:.2f}")
print(f"Validation Accuracy: {hard_val_acc:.2f}")

print('\n')

print("=== Soft Margin SVM (C=1.0) ===")
soft_svm = LinearSVM(C=1.0)
soft_svm.fit(X_train, y_train)
soft_train_acc = np.mean(soft_svm.predict(X_train) == y_train)
soft_val_acc = np.mean(soft_svm.predict(X_val) == y_val)
print(f"Training Accuracy: {soft_train_acc:.2f}")
print(f"Validation Accuracy: {soft_val_acc:.2f}")

# 测试集结果
hard_test_acc = np.mean(hard_svm.predict(X_test) == y_test)
soft_test_acc = np.mean(soft_svm.predict(X_test) == y_test)
print("\n=== Test Accuracy ===")
print(f"Hard Margin: {hard_test_acc:.2f}")
print(f"Soft Margin (C=1.0): {soft_test_acc:.2f}")

prediction_hard=hard_svm.predict(X_test)
prediction_soft=soft_svm.predict(X_test)
#可视化混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#混淆矩阵:硬间隔SVM测试集
conf_matrix = confusion_matrix(y_test, prediction_hard, labels=[1, -1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Malignant (1)", "Benign (-1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("CM - Hard_SVM_test")
plt.show()

#混淆矩阵:软间隔SVM测试集
conf_matrix = confusion_matrix(y_test, prediction_soft, labels=[1, -1])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Malignant (1)", "Benign (-1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("CM - Soft_SVM_test")
plt.show()







