# 2.0版 旨在尽量减少数据和逻辑耦合 增强模块化和可维护性
import sys
import random
import pandas as pd
import numpy as np

# 引入 PyQt5 相关
from PyQt5 import QtCore,QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *

# 引入自定义模块
from utils import *

# 程序窗口     
class ApplicationWindow(QMainWindow):
    def __init__(self):

        # 程序窗口初始化
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.center()
        self.setWindowTitle("拉曼光谱分析")
        self.setWindowIcon(QIcon('icon.jpg'))   
        
        # 全局变量
        self.classType = 'binary' # 分类模式 默认二分类
        self.classNum = 2         # 分类数目 默认是二分类模式下的 2
        self.threshold = 20       # 分类阈值 默认 20
        self.components = 3       # 主成分数 默认 3
        self.scross = 10          # S折交叉验证 默认 10
        self.X = None             # 原始数据
        self.headline = None      # 原始表头 即浓度
        self.newX = None          # 降维后的数据
        self.labels = None        # 标注序列
        self.ratio = None         # 方差贡献率
        self.OSVM = None          # 最佳 SVM 模型
        self.pcaModel = None      # pca 模型

        # 布局
        self.main_widget = QWidget(self)
        mlayout = QHBoxLayout(self.main_widget)
        llayout = QFormLayout()
        rlayout = QVBoxLayout()
        
        addForm(self, llayout)    # 表单初始化
        addTable(self, rlayout)  # 表格初始化
        self.addMenu()                         # 菜单栏初始化

        # 窗口初始化
        mlayout.addLayout(llayout)
        mlayout.addLayout(rlayout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    # 居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    # 菜单栏初始化   
    def addMenu(self):
        self.file_menu = QMenu('文件', self)
        self.file_menu.addAction('导入训练集', lambda: openTrain(self))
        self.file_menu.addAction('导入模型', lambda: openModel(self))
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QMenu('帮助', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('关于', self.about)
    def fileQuit(self):
        self.close()
    def closeEvent(self, ce):
        self.fileQuit()
    def about(self):
        QMessageBox.about(self, "关于",
        """
        对拉曼光谱数据进行分类的支持向量机模型
        将拉曼光谱数据根据浓度分类并进行预测
        
        作者：CheeReus_11
        """
        )

# 文件操作

# csv文件读取 文件中第一行为浓度
# 输出 纯数据矩阵 X 、浓度向量 y 其中浓度向量去除了小数点
def csvReader(filePath):
    
    # 由于路径中的中文会导致 read_csv 报错，因此先用 open 打开
    f = open(filePath,'r')
    data=pd.read_csv(f,header=None, low_memory=False)
    f.close()
    array_data = np.array(data)
    X=array_data.T[1:,1:]  # 纯数据
    y=array_data.T[1:,0]   # 第一行
    y = y * 100            # 支持向量机训练仅支持 int 类型的 label 因此将浓度值进行去除小数点的处理
    return X,y

def openTrain(self):

    filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "CSV Files (*.csv)")
    if filePath:
        showLabel(self.filepath, "已选择训练集：" + filePath)
        self.X, self.headline = csvReader(filePath)
        self.labels, self.classNum = getLabel(self)
        setTable(self, self.labels, self.X)
        self.btn0.setEnabled(True)
        self.btn1.setEnabled(True)
        self.btn2.setEnabled(True)
        self.btn3.setEnabled(True)
        self.btn4.setEnabled(True)
        
    else:
        QMessageBox.warning(self,"温馨提示","打开文件错误，请重新尝试！",QMessageBox.Cancel)

    # 导入训练集后才允许修改参数
    self.btn0.setEnabled(True)
    self.btn1.setEnabled(True)
    self.btn2.setEnabled(True)
    self.btn3.setEnabled(True)

# 读取模型
def openModel(self):
    filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "model Files (*.pkl)")
    if filePath:
        self.OSVM, self.pcaModel, self.classNum, self.threshold, self.components, self.scross = modelReader(filePath)

        self.thresholdLabel.setText('二分类阈值：' + str(self.threshold))
        self.componentsLabel.setText('主成分数目：' + str(self.components))
        self.scrossLabel.setText('交叉验证数目：' + str(self.scross))
        self.filepath.setText("已选择模型：" + filePath)
        if self.classNum > 2:
            showLabel(self.classTypeLabel,'当前模式：多分类(' + str(self.classNum) + '类)')
        else:
            showLabel(self.classTypeLabel,'当前模式：二分类')
        self.btn6.setVisible(True)
        # 导入模型后不允许修改参数及主成分分析、训练等操作
        self.btn0.setEnabled(False)
        self.btn1.setEnabled(False)
        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)
        self.btn4.setEnabled(False)
        self.btn5.setEnabled(False)
        self.ratioLabel.setVisible(False)
        self.btn5.setEnabled(False)
        self.svmLabel.setVisible(False)
        
    else:
        QMessageBox.warning(self,"温馨提示","打开文件错误，请重新尝试！",QMessageBox.Cancel)

# 表单操作

# 表单初始化
def addForm(self,llayout):

    self.filepath = QLabel('请先导入数据或模型！')
    self.filepath.setFixedWidth(600)
    llayout.addRow(self.filepath)

    # 参数展示及修改
    self.classTypeLabel = QLabel('当前模式：二分类')
    self.thresholdLabel = QLabel('二分类阈值：' + str(self.threshold))
    self.componentsLabel = QLabel('主成分数目：' + str(self.components))
    self.scrossLabel = QLabel('交叉验证折数：' + str(self.scross))
    self.btn0 = QPushButton('切换')
    self.btn1 = QPushButton('修改')
    self.btn2 = QPushButton('修改')
    self.btn3 = QPushButton('修改')
    self.btn0.clicked.connect(lambda: toggleClassType(self))
    self.btn1.clicked.connect(lambda: changeParas(self,1))
    self.btn2.clicked.connect(lambda: changeParas(self,2))
    self.btn3.clicked.connect(lambda: changeParas(self,3))
    llayout.setLabelAlignment(QtCore.Qt.AlignRight) # 标签右对齐
    llayout.addRow(self.classTypeLabel,self.btn0)
    llayout.addRow(self.thresholdLabel,self.btn1)
    llayout.addRow(self.componentsLabel,self.btn2)
    llayout.addRow(self.scrossLabel,self.btn3)

    # 导入训练集后才允许修改参数
    self.btn0.setEnabled(False)
    self.btn1.setEnabled(False)
    self.btn2.setEnabled(False)
    self.btn3.setEnabled(False)

    # 主成分分析
    self.btn4 = QPushButton('主成分分析')
    self.btn4.clicked.connect(lambda: pca(self, self.X, self.components, showTable=True))
    llayout.addRow(self.btn4)
    self.ratioLabel = QLabel('方差贡献率：')
    llayout.addRow(self.ratioLabel)
    self.btn4.setEnabled(False)
    self.ratioLabel.setVisible(False)

    # 训练支持向量机
    self.btn5 = QPushButton('训练支持向量机')
    self.btn5.clicked.connect(lambda: getSVM(self, self.newX, self.labels, self.scross))
    llayout.addRow(self.btn5)
    self.svmLabel = QLabel('交叉验证准确率：')
    llayout.addRow(self.svmLabel)
    self.btn5.setEnabled(False)
    self.svmLabel.setVisible(False)

    # 预测
    self.btn6 = QPushButton('开始预测')
    self.btn6.clicked.connect(lambda: getPredict(self))
    self.predictLabel = QLabel('预测结果已保存至 output/predictResult.csv')
    llayout.addRow(self.btn6)
    llayout.addRow(self.predictLabel)
    self.btn6.setVisible(False)
    self.predictLabel.setVisible(False)

    # 清空表格
    self.btn7 = QPushButton('清空右侧表格')
    self.btn7.clicked.connect(lambda: TableOperator.clearTable(self))
    llayout.addRow(self.btn7)
    self.btn7.setEnabled(False)

# 设置并展示label文字
def showLabel(labelObj, labelTxt):
    labelObj.setText(labelTxt)
    labelObj.setVisible(True)

# 表格操作

# 表格初始化
def addTable(self,rlayout):
    self.tableWidget = QTableWidget()
    self.tableWidget.setRowCount(20)
    self.tableWidget.setColumnCount(20)
    self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(20)])
    rlayout.addWidget(self.tableWidget)

# 清空表格
def clearTable(self):
    self.tableWidget.setRowCount(20)
    self.tableWidget.setColumnCount(20)
    self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(20)])
    for i in range(20):
        for j in range(20):
            #为每个表格内添加空白数据
            self.tableWidget.setItem(i,j,QTableWidgetItem(''))
    # 关闭清空数据按钮
    self.btn7.setEnabled(False)

# 表格数据展示 与第一版区别主要在 此处不再处理label 只关心数据填充
def setTable(self,head,data):
    data = data.T
    rows, columns = data.shape
    if head is None:
        head = range(rows)
    self.tableWidget.setRowCount(rows+1)
    self.tableWidget.setColumnCount(columns)
    self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(rows+1)])
    for k in range(columns):
        #为表格添加内置头部
        self.tableWidget.setItem(0,k,QTableWidgetItem(str(head[k])))
    for i in range(rows):
        for j in range(columns):
            #为每个表格内添加数据
            self.tableWidget.setItem(i+1,j,QTableWidgetItem(str(data[i,j])))
    # 开启清空数据按钮
    self.btn7.setEnabled(True)

# 数据预处理及参数配置

# 获取分类标签及分类数目 二分类时按阈值返回0或1 多分类时原样返回
# 同时返回分类数目 并更新界面文字
def getLabel(self):

    if self.classType == 'multi':
        nums = len(pd.value_counts(self.headline))
        showLabel(self.classTypeLabel,'当前模式：多分类(' + str(nums) + '类)')
        return self.headline, nums
    else:
        headSize = len(self.headline)
        labels = np.zeros(headSize)
        for i in range(headSize):
            if self.headline[i] > (self.threshold * 100):
                labels[i] = 1   
        showLabel(self.classTypeLabel,'当前模式：二分类')      
        return labels, 2

# 切换分类模式
def toggleClassType(self):
    if self.classType == 'binary':
        self.classType = 'multi'
    else:
        self.classType = 'binary'
    self.labels, self.classNum = getLabel(self)
    if self.classType == 'binary':
        setTable(self, self.labels, self.X)
    else:
        setTable(self, self.labels / 100, self.X)
    # 切换后需要重新训练 因此关闭部分按钮和文字
    self.ratioLabel.setVisible(False)
    self.btn5.setEnabled(False)
    self.svmLabel.setVisible(False)
    self.btn6.setVisible(False)
    self.predictLabel.setVisible(False)

# 参数设定
def changeParas(self,type):
    if type == 1:
        self.threshold, ok = QInputDialog.getDouble(self, "二分类阈值", "请输入阈值:", 25.00, 0, 100, 2)
        self.thresholdLabel.setText('二分类阈值：' + str(self.threshold))
    if type == 2:
        self.components, ok = QInputDialog.getInt(self, "主成分数目", "请输入整数:", 3, 2, 1000, 0)
        self.componentsLabel.setText('主成分数目：' + str(self.components))
    if type == 3:
        self.scross, ok = QInputDialog.getInt(self, "交叉验证数目", "请输入整数:", 10, 2, 20, 0)
        self.scrossLabel.setText('交叉验证数目：' + str(self.scross))

# 机器学习模块

# 主成分分析
def pca(self,X,n, showTable):
    self.newX, self.ratio, self.pcaModel = pca_op(X,n)
    ratioText = "方差贡献率：\n"
    sum = 0
    for i in range(len(self.ratio)):
        ratioText = ratioText + str(self.ratio[i]) + "\n"
        sum = sum + self.ratio[i]
    ratioText = ratioText + '\n总贡献率：' + str(sum) + "\n降维后数据及标注序列已保存至 output 目录下"
    showLabel(self.ratioLabel,ratioText)
    if showTable:
        setTable(self, self.labels, self.newX)
    self.btn5.setEnabled(True)

# 支持向量机训练
def getSVM(self,newX,labels,s):
    self.OSVM, scores, bestParas = cross_validation(newX,labels,s)
    modelSave(self.OSVM, self.pcaModel, self.classNum, self.threshold, self.components, self.scross)

    # 如果要针对不同的核函数加 if else 就改这个 bestText
    bestText = "最优参数：\n" + "C:" + str(bestParas['C']) + '\ngamma:' + str(bestParas['gamma']) + '\ndegree:' + str(bestParas['degree']) + '\nkernel:' + str(bestParas['kernel']) + '\ndecision_function_shape:' + str(bestParas['decision_function_shape'])

    svmText = "\n交叉验证准确率：\n"
    avg = 0
    for i in range(len(scores)):
        avg = avg + scores[i]
        svmText = svmText + str(scores[i]) + "\n"
    avg = avg / len(scores)
    svmText = bestText + svmText + '平均准确率：' + str(avg) + "\n模型已保存至 output/svm_model_with_pca.pkl"
    showLabel(self.svmLabel,svmText)
    self.btn6.setVisible(True)

# 预测
def getPredict(self):
    # 开始预测后不允许修改参数及主成分分析、训练等操作，除非重新导入训练集
    self.btn0.setEnabled(False)
    self.btn1.setEnabled(False)
    self.btn2.setEnabled(False)
    self.btn3.setEnabled(False)
    self.btn4.setEnabled(False)
    self.btn5.setEnabled(False)
    filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "CSV Files (*.csv)")
    if filePath:
        self.filepath.setText("已选测试集：" + filePath)
        self.X, self.headline = csvReader(filePath)
        self.newX = re_pca(self.X, self.pcaModel)
        self.labels = self.OSVM.predict(self.newX)
        
        correct = 0
        if self.classType == 'binary':
            labelY, nums = getLabel(self)
            for i in range(len(self.labels)):
                if self.labels[i] == labelY[i]:
                    correct = correct + 1
        else:
            for i in range(len(self.labels)):
                if self.labels[i] == self.headline[i]:
                    correct = correct + 1
        correctRate = correct / len(self.labels)
        
        if self.classType == 'multi':
            self.labels = self.labels / 100
        setTable(self, self.labels, self.newX)
        data = np.hstack((self.labels[:,None], self.X))
        data = np.hstack(((self.headline / 100)[:,None], data))
        pd.DataFrame.to_csv(pd.DataFrame(data.T),'output/predictResult.csv', mode='w',header=None,index=None)
        showLabel(self.predictLabel,'准确率：' + str(correctRate) + '\n预测结果已保存至 output/predictResult.csv')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.showMaximized()
    #sys.exit(qApp.exec_())
    app.exec_()
