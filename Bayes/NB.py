######一、朴素贝叶斯分类器
import numpy as np

class NBayes(object):

    # 设置属性
    def __init__(self):
        self.trainSet = 0  # 训练集数据
        self.trainLabel = 0  # 训练集标记
        self.yProba = {}  # 先验概率容器
        self.xyProba = {}  # 条件概率容器
        self.ySet = {}  # 标记类别对应的数量
        self.ls = 1  # 加入的拉普拉斯平滑的系数
        self.n_samples = 0  # 训练集样本数量
        self.n_features = 0  # 训练集特征数量

    # 计算P(y)先验概率
    def calPy(self, y, LS=True):
        """
        计算先验概率，也就是每个标记的占比

        Parameters
        ----------
        y : 1D array-like
            trainLabel.
        LS : bool, optional
            Weather Laplace Smoothing. The default is True.

        Returns
        -------
        None.

        """
        Py = {}
        yi = {}
        ySet = np.unique(y)
        for i in ySet:
            Py[i] = (sum(y == i) + self.ls) / (self.n_samples + len(ySet))
            yi[i] = sum(y == i)
        self.yProba = Py
        self.ySet = yi
        return

    # 计算P(x|y)条件概率
    def calPxy(self, X, y, LS=True):
        """
        计算先验概率，也就是每类分类中，每个变量值的占比

        Parameters
        ----------
        X : 2D array-like
            trainSet.
        y : 1D array-like
            trainLabel.
        LS : bool, optional
            Weather Laplace Smoothing. The default is True.

        Returns
        -------
        None.

        """
        Pxy = {}
        for yi, yiCount in self.ySet.items():
            Pxy[yi] = {}  # 第一层是标签Y的分类
            for xIdx in range(self.n_features):
                Pxy[yi][xIdx] = {}  # 第二层是不同的特征
                # 下标为第xIdx的特征数据
                Xi = X[:, xIdx]
                XiSet = np.unique(Xi)
                XiSetCount = XiSet.size
                # 下标为第xIdx，并标签为yi的特征数据
                Xiyi = X[np.nonzero(y == yi)[0], xIdx]
                for xi in XiSet:
                    Pxy[yi][xIdx][xi] = self.classifyProba(xi, Xiyi, XiSetCount)  # 第三层是变量Xi的分类概率，离散变量
        self.xyProba = Pxy
        return

    # 离散变量直接计算概率
    def classifyProba(self, x, xArr, XiSetCount):
        Pxy = (sum(xArr == x) + self.ls) / (xArr.size + XiSetCount)  # 加入拉普拉斯修正的概率
        return Pxy

    def calContinuousParams(self, X, y):
        """
        计算每个连续特征在每个类别下的均值和方差。

        Parameters
        ----------
        X : 2D array-like
            trainSet.
        y : 1D array-like
            trainLabel.

        Returns
        -------
        None.
        """
        continuousParams = {}
        for yi in self.ySet.keys():  # 遍历每个标签类别
            continuousParams[yi] = {}
            for xIdx in range(self.n_features):  # 遍历每个特征
                # 筛选出标签为 yi 的样本中第 xIdx 个特征的值
                Xiyi = X[np.nonzero(y == yi)[0], xIdx]
                # 计算均值和方差
                mu = np.mean(Xiyi)
                sigma2 = np.var(Xiyi)
                continuousParams[yi][xIdx] = (mu, sigma2)
        self.continuousParams = continuousParams
        return

    def gaussianProba(self, x, mu, sigma2):
        """
        计算高斯分布概率密度值。

        Parameters
        ----------
        x : float
            特征值.
        mu : float
            均值.
        sigma2 : float
            方差.

        Returns
        -------
        float
            概率密度值.

        """
        return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

    # 训练
    def train(self, X, y, featureTypes):
        """
        训练方法，支持离散和连续特征。

        Parameters
        ----------
        X : 2D array-like
            训练集.
        y : 1D array-like
            训练标签.
        featureTypes : list of str
            每个特征的类型 ("discrete" 或 "continuous").

        Returns
        -------
        None.
        """
        self.n_samples, self.n_features = X.shape
        self.featureTypes = featureTypes
        # 计算先验概率
        self.calPy(y)
        print('P(y)训练完毕!')
        # 处理特征
        for idx, fType in enumerate(featureTypes):
            if fType == "discrete":
                self.calPxy(X, y)
            elif fType == "continuous":
                self.calContinuousParams(X, y)
        print('P(x|y)训练完毕!')
        self.trainSet = X
        self.trainLabel = y
        return

    # 预测
    def predict(self, X):
        m, n = X.shape
        proba = np.zeros((m, len(self.yProba)))
        for i in range(m):
            for idx, (yi, Py) in enumerate(self.yProba.items()):
                proba_idx = Py
                for xIdx in range(n):
                    xi = X[i, xIdx]
                    if self.featureTypes[xIdx] == "discrete":  # 离散特征
                        proba_idx *= self.xyProba[yi][xIdx].get(xi, 1e-6)  # 防止KeyError
                    elif self.featureTypes[xIdx] == "continuous":  # 连续特征
                        mu, sigma2 = self.continuousParams[yi][xIdx]
                        proba_idx *= self.gaussianProba(xi, mu, sigma2)
                proba[i, idx] = proba_idx
        return proba


