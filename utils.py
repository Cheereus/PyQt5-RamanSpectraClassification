import pandas as pd
import numpy as np
from sklearn.decomposition import PCA    # 导入PCA模块
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer    # 导入数据预处理归一化类
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

# 输入数据矩阵 X 及主成分数目 c
# 输出降维后的数据及贡献率
def pca_op(X,c=3):

    prepress = Normalizer()  #
    x = prepress.fit_transform(X)  # 拟合转换数据一统一量纲标准化
    pca_result = PCA(n_components=c)      # 降维后有c个主成分
    pca_result.fit(x)                     # 训练
    newX=pca_result.fit_transform(x)      # 降维后的数据

    # 保存为csv文件
    np.savetxt('output/pca_x.csv', newX, delimiter = ',')

    return newX, pca_result.explained_variance_ratio_

# 输入降维后的数据 x 标注 y 交叉验证折数 s
# 输出最佳的SVM模型
def cross_validation(x,y,s=10):

    clf = svm.SVC(kernel='rbf', verbose=True,gamma='scale')
    scores = cross_val_score(clf, x, y.ravel(), cv=s)
    clf.fit(x, y)
    joblib.dump(clf, 'output/svm_model.pkl')
    return clf, scores