# PyQt5-RamanSpectraClassification

Raman spectral classification base on PyQt5

基于 PyQt5 的拉曼光谱分类 GUI [查看中文版README](https://github.com/Cheereus/PyQt5-RamanSpectraClassification/blob/master/README-zh.md)

version: 2.0

&copy; This project belongs to my senior's Master thesis《拉曼光谱结合化学计量学在磷矿检测中的应用》(2020)

If this project is useful to you, please give me a star !

## Main Function

### Import Training Set

Training set file format requirements：

* It should be `.csv` file.
* The first row is the concentration mark, and it will be divided into high and low categories according to the classification threshold in the binary classification.
* The first column is the spectral wavelength, which has no specific meaning in our data analysis.

### Parameter Configuration

Classification mode：

* Binary classification. According to the classification threshold, the concentration markers are divided into high and low categories.
* Multi-class classification. datasets will be directly divided into the original categories by concentration mark.
* Display the number of categories.

Number of principal components:

* It can be customized.
* 3 by default.

Cross-validation folds：

* It can be customized.
* 10 by default.

### Principal component analysis (PCA)

Perform dimensionality reduction according to the number of principal components configured, and output the variance contribution rate of each dimension and their sum.

### Support vector machine (SVM) training

Use grid search to find the optimal parameters and display them.

* The labels are 0 and 1 in binary classification.
* In multi-class classification, the labels are concentration * 100 (so that the decimal point is removed), and convert them into int type.
* Perform cross-validation according to the optimal parameters, and output the accuracy rates and average accuracy rate of the cross-validation.

### Save and import SVM model file

The model of the above steps can be saved and imported as file, eliminating the need for repeated steps of retraining

### Prediction

Import the data set that needs to be predicted then make prediction:

* The reading method of the prediction set is the same as that of the training set, so it is necessary to ensure that there is no data in the first row and first column, because this part of the content will be discarded.
* The results will alse be saved as `.csv` file.

### Data demonstration

Each step will instantly display the output data of the current step in the table on the right.

## Instructions for use

### Running environment

Anaconda (Python 3.7.1 64-bit)

Windows 10 Pro 64-bit

Command line start:

```shell
python main.py
```

### Dependencies

pyqt5

pandas

numpy

sklearn

### File instructions

The code contains a lot of comments to facilitate readers’ understanding.

#### main.py

Core functions including：

* GUI initialization and main process.
* `.csv` file operations and label treatment.
* Table display and processing.
* Parameter configuration and button operations.

#### utils.py

Mainly includes machine learning fuctions：

* PCA
* SVM training including cross-validation.
* Save and import model file.

### Project achievements

The entire process GUI of spectral classification processing based on PyQt5 including：

* File import, text display, button events, layout and typesetting.
* QTableWidget table display.

Support vector machine model training and prediction based on sklearn.

File access, data statistics and analysis processing.

### Interface language

The interface language is `Chinese`, if you need a English version (or you want to provide a English version) please send me a email (fanwei1995@hotmail.com) or create an issue.
