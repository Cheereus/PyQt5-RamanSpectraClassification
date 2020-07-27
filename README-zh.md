# PyQt5-RamanSpectraClassification

基于 PyQt5 的拉曼光谱分类 GUI

version: 2.0

&copy; 本工作属于学长的硕士毕业论文《拉曼光谱结合化学计量学在磷矿检测中的应用》(2020)

如果本工作对您有帮助，请留下一个star，万分感激！

## 主要功能

### 导入训练集

所需的训练集格式要求：

* 要求为 csv 文件
* 第一行为浓度标记，二分类时将根据分类阈值分为高低两类
* 第一列为光谱波长，在分析时无具体意义

### 参数配置

分类模式：

* 二分类，根据分类阈值将浓度标记分为高低两类
* 多分类，直接按浓度标记分为原本的类目
* 展示分类数目

主成分数目：

* 可自定义
* 默认 3 种

交叉验证折数：

* 可自定义
* 默认 10 折

### 主成分分析

使用 PCA 方法，根据配置的主成分数目进行降维，并输出每维的方差贡献率及其总和

### 支持向量机训练

使用网格搜索来寻找最优参数，并展示

* 二分类时标签为 0,1
* 多分类时为 浓度 * 100 即去除了小数点，并转为 int 类型
* 根据最优参数进行交叉验证，并输出交叉验证的准确率及平均准确率

### 保存和导入 SVM 模型文件

可以保存和导入上述步骤的模型文件，免去重新训练的重复步骤

### 预测

导入需要预测的数据集进行预测：

* 预测集的读取方法和训练集相同，因此需要保证第一行第一列无数据，因为这部分内容会被丢弃
* 预测结果也会保存至 csv 文件中

### 数据展示

每步操作都会即时在右侧表格中展示当前步骤的输出数据

## 使用说明

### 运行环境

Anaconda (Python 3.7.1 64-bit)

Windows 10 Pro 64-bit

命令行启动:

```shell
python main.py
```

### 涉及依赖

pyqt5

pandas

numpy

sklearn

### 文件说明

代码中包含了相当多的注释，方便读者理解

#### main.py

核心功能，涉及：

* GUI 初始化及主要流程
* csv 文件读取及标签处理
* 表格展示及处理
* 参数配置、按钮操作

#### utils.py

主要是涉及机器学习的内容：

* 主成分分析
* 支持向量机训练，包括交叉验证、模型保存
* 模型读取

### 项目成果

实现了基于 pyqt5 的光谱分类处理的整个流程 GUI 包括：

* 文件导入、文本展示、按钮事件、布局排版
* QTableWidget 表格展示

实现了基于 sklearn 的支持向量机模型训练及预测

实现了文件存取，数据统计及分析处理

### 总结

是第一次使用 pyqt5 制作 GUI，之前用过 Electron

python 有天然的功能优势，但在界面布局和美化上，难以与基于 H5 的 Electron 相比

踩了很多的坑，也对 python 和涉及到的各种库有了更多的了解

参考了大量的资料，无法再一一列举，在此对所有在网络上辛勤奉献的同行们表示由衷的感激