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

# 数据处理
class DataOperator:

    # 切换模式
    def toggleClassType(self):
        if self.classType == 'binary':
            self.classType = 'multi'
        else:
            self.classType = 'binary'
        DataOperator.updateLabels(self)

    # 更新分类数目
    def updateLabels(self):
        if self.classType == 'binary':
            self.classTypeLabel.setText('当前模式：二分类')
            self.classNumberLabel.setText('当前训练集类别数目：2')
            self.labels = DataOperator.getLabel(self.headline,self.threshold)
        else:
            self.classTypeLabel.setText('当前模式：多分类')
            self.classNumberLabel.setText('当前训练集类别数目：' + str(self.classNum))
            self.labels = self.headline
        print(self.labels)
        

    # csv文件处理 文件中第一行为浓度
    # 输出 纯数据矩阵 X 、浓度向量 y
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

    # 根据设定的阈值来获取标注向量 labels，低于阈值0 高于阈值1
    def getLabel(y,threshold):
        size = len(y)
        labels = np.zeros(size)
        for i in range(size):
            if y[i] > (threshold * 100):
                labels[i] = 1               
        np.savetxt('output/labels.csv', labels, delimiter = ',')
        return labels

    # 表格数据展示
    def setTable(self,head,data):
        data = data.T
        rows, columns = data.shape
        headLabel = []
        if head is None:
            head = range(rows)

        if self.classType == 'binary':
            for k in range(len(head)):
                if(head[k] < 0.5):
                    headLabel.append("低浓度")
                else:
                    headLabel.append("高浓度")
        else:
            headLabel = head / 100 # 将浓度值恢复小数点
        
        self.tableWidget.setRowCount(rows+1)
        self.tableWidget.setColumnCount(columns)
        self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(rows+1)])
        for k in range(columns):
            #为表格添加内置头部
            self.tableWidget.setItem(0,k,QTableWidgetItem(str(headLabel[k])))
        for i in range(rows):
            for j in range(columns):
                #为每个表格内添加数据
                self.tableWidget.setItem(i+1,j,QTableWidgetItem(str(data[i,j])))
        
        self.btn7.setEnabled(True)
    
    # 清空表格内容
    def clearTable(self):
        self.tableWidget.setRowCount(20)
        self.tableWidget.setColumnCount(20)
        self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(20)])
        for i in range(20):
            for j in range(20):
                #为每个表格内添加数据
                self.tableWidget.setItem(i,j,QTableWidgetItem(''))
        

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

    # 主成分分析
    def pca(self,X,n):
        self.newX, self.ratio = pca_op(X,n)
        ratioText = "方差贡献率：\n"
        sum = 0
        for i in range(len(self.ratio)):
            ratioText = ratioText + str(self.ratio[i]) + "\n"
            sum = sum + self.ratio[i]
        ratioText = ratioText + '\n总贡献率：' + str(sum) + "\n降维后数据及标注序列已保存至 output 目录下"
        DataOperator.setTable(self, self.labels, self.newX)
        self.ratioLabel.setText(ratioText)
        self.ratioLabel.setVisible(True)
        self.btn5.setEnabled(True)
    
    # 支持向量机训练
    def getSVM(self,newX,labels,s):
        self.OSVM, scores = cross_validation(newX,labels,s)
        svmText = "交叉验证准确率：\n"
        for i in range(len(scores)):
            svmText = svmText + str(scores[i]) + "\n"
        svmText = svmText + "模型已保存至 output/svm_model.pkl"
        self.svmLabel.setText(svmText)
        self.svmLabel.setVisible(True)
        self.btn6.setVisible(True)

    # 预测
    def getPredict(self):
        filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "CSV Files (*.csv)")
        if filePath:
            self.filepath.setText("已选测试集：" + filePath)
            self.X, y = DataOperator.csvReader(filePath)
            DataOperator.pca(self, self.X, self.components)
            self.labels = self.OSVM.predict(self.newX)
            DataOperator.setTable(self, self.labels, self.newX)

            data = np.hstack((self.labels[:,None], self.X))
            pd.DataFrame.to_csv(pd.DataFrame(data.T),'output/predictResult.csv', mode='w',header=None,index=None)
            self.predictLabel.setVisible(True)
            
        else:
            QMessageBox.warning(self,"温馨提示","打开文件错误，请重新尝试！",QMessageBox.Cancel)

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

        # 布局
        self.main_widget = QWidget(self)
        mlayout = QHBoxLayout(self.main_widget)
        llayout = QFormLayout()
        rlayout = QVBoxLayout()

        # 表单初始化
        self.addForm(llayout)

        # 表格初始化
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(20)
        self.tableWidget.setColumnCount(20)
        self.tableWidget.setVerticalHeaderLabels([str(item) for item in range(20)])
        rlayout.addWidget(self.tableWidget)

        # 菜单栏初始化
        self.file_menu = QMenu('文件', self)
        self.file_menu.addAction('导入训练集', self.trainOpen)
        self.file_menu.addAction('导入模型', self.modelOpen)
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QMenu('帮助', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('关于', self.about)
        
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

    # 读取训练集
    def trainOpen(self, type='binary'):
        filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "CSV Files (*.csv)")
        if filePath:
            self.filepath.setText("已选择文件：" + filePath)
            self.X, self.headline = DataOperator.csvReader(filePath)
            print(pd.value_counts(self.headline))
            self.classNum = len(pd.value_counts(self.headline))
            DataOperator.updateLabels(self)
            DataOperator.setTable(self, self.labels, self.X)
            self.btn4.setEnabled(True)
            
        else:
            QMessageBox.warning(self,"温馨提示","打开文件错误，请重新尝试！",QMessageBox.Cancel)

    # 读取模型
    def modelOpen(self):
        filePath,filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "model Files (*.pkl)")
        if filePath:
            self.OSVM = modelReader(filePath)
            self.filepath.setText("已选择模型：" + filePath)
            self.btn6.setVisible(True)
            
        else:
            QMessageBox.warning(self,"温馨提示","打开文件错误，请重新尝试！",QMessageBox.Cancel)


    # 表单生成
    # TODO 采用了非常粗暴的逐项添加，需要采用更优的方法
    def addForm(self,llayout):

        self.filepath = QLabel('尚未选择文件！')
        self.filepath.setFixedWidth(600)
        llayout.addRow(self.filepath)
        self.classNumberLabel = QLabel('当前训练集类别数目：2')
        llayout.addRow(self.classNumberLabel)

        # 参数展示及修改
        self.classTypeLabel = QLabel('当前模式：二分类')
        self.thresholdLabel = QLabel('二分类阈值：' + str(self.threshold))
        self.componentsLabel = QLabel('主成分数目：' + str(self.components))
        self.scrossLabel = QLabel('交叉验证折数：' + str(self.scross))
        btn0 = QPushButton('切换')
        btn1 = QPushButton('修改')
        btn2 = QPushButton('修改')
        btn3 = QPushButton('修改')
        btn0.clicked.connect(lambda: DataOperator.toggleClassType(self))
        btn1.clicked.connect(lambda: DataOperator.changeParas(self,1))
        btn2.clicked.connect(lambda: DataOperator.changeParas(self,2))
        btn3.clicked.connect(lambda: DataOperator.changeParas(self,3))
        llayout.setLabelAlignment(QtCore.Qt.AlignRight) # 标签右对齐
        llayout.addRow(self.classTypeLabel,btn0)
        llayout.addRow(self.thresholdLabel,btn1)
        llayout.addRow(self.componentsLabel,btn2)
        llayout.addRow(self.scrossLabel,btn3)

        # 主成分分析
        self.btn4 = QPushButton('主成分分析')
        self.btn4.clicked.connect(lambda: DataOperator.pca(self, self.X, self.components))
        llayout.addRow(self.btn4)
        self.ratioLabel = QLabel('方差贡献率：')
        llayout.addRow(self.ratioLabel)
        self.btn4.setEnabled(False)
        self.ratioLabel.setVisible(False)

        # 训练支持向量机
        self.btn5 = QPushButton('训练支持向量机')
        self.btn5.clicked.connect(lambda: DataOperator.getSVM(self, self.newX, self.labels, self.scross))
        llayout.addRow(self.btn5)
        self.svmLabel = QLabel('交叉验证准确率：')
        llayout.addRow(self.svmLabel)
        self.btn5.setEnabled(False)
        self.svmLabel.setVisible(False)

        # 预测
        self.btn6 = QPushButton('开始预测')
        self.btn6.clicked.connect(lambda: DataOperator.getPredict(self))
        self.predictLabel = QLabel('预测结果已保存至 output/predictResult.csv')
        llayout.addRow(self.btn6)
        llayout.addRow(self.predictLabel)
        self.btn6.setVisible(False)
        self.predictLabel.setVisible(False)

        # 清空表格
        self.btn7 = QPushButton('清空右侧表格')
        self.btn7.clicked.connect(lambda: DataOperator.clearTable(self))
        llayout.addRow(self.btn7)
        self.btn7.setEnabled(False)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.showMaximized()
    #sys.exit(qApp.exec_())
    app.exec_()
