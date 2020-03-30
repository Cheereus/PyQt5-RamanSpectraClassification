import pandas as pd
import numpy as np
from sklearn.decomposition import PCA    # 导入PCA模块
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer    # 导入数据预处理归一化类
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

# 输入数据矩阵 X 及主成分数目 c
# 主成分数目 c 要小于数据矩阵 X 的长和宽 即 c <= min(x.shape[0],x.shape[1])
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
# 交叉验证折数是有限制的 必须保证训练集每个类都能至少分为 s 份 即单个类的数目 sn 应满足 sn / s >= 1
# 输出最佳的SVM模型
def cross_validation(x,y,s=10):

    svc = svm.SVC()
    parameters = [
        {
            'C': [1, 3, 5],
            'gamma': [0.001, 0.1, 1, 10],
            'degree': [3,5,7,9],
            'kernel': ['linear','poly', 'rbf', 'sigmoid'],
            'decision_function_shape': ['ovo', 'ovr' ,None]
        }
    ]
    clf=GridSearchCV(svc,parameters,cv=s,refit=True)
    y = y.astype('int')
    clf.fit(x, y)
    print(clf.best_params_)
    print(clf.best_score_)
    joblib.dump(clf.best_estimator_, 'output/svm_model.pkl')

    cross_model = svm.SVC(C=clf.best_params_['C'],degree=clf.best_params_['degree'],kernel=clf.best_params_['kernel'],gamma=clf.best_params_['gamma'], decision_function_shape=clf.best_params_['decision_function_shape'], verbose=0)
    scores = cross_val_score(cross_model, x, y.ravel(), cv=s)

    return clf.best_estimator_, scores, clf.best_params_

# 读取模型
def modelReader(filePath):
        
    f = open(filePath,'rb')
    model = joblib.load(f)
    f.close()
    return model

# 近红外光谱基线校正 airPLS

def airPLS():
    
    return None