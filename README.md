# Guwen-UNILM: Machine Translation Between Ancient and Modern Chinese Based on Pre-Trained Models
## Introduction
<div  align="center">   
 <img src="https://user-images.githubusercontent.com/30574139/163604086-13213a39-ab35-42d1-806f-04f48914b6f3.png" width = "600" height = "400" alt="图片名称" align=center />
</div> 

本仓库是基于[bert4keras](https://github.com/bojone/bert4keras)实现的古文-现代文翻译模型。具体使用了基于掩码自注意力机制的[UNILM(Li al., 2019)](https://arxiv.org/abs/1905.03197)预训练模型作为翻译系统的backbone。我们首先使用了普通的中文（现代文）BERT、Roberta权重作为UNILM的初始权重以训练UNILM模型（具体在文中分别为B-UNILM以及R-UNILM）。为了更好的使UNILM模型适应古文的特性，我们尝试使用了在古文预训练模型[Guwen-BERT](https://github.com/Ethan-yt/guwenbert)，作为UNILM的初始权重，并且获得了最优的效果。除此之外，我们研究了分词对于古文-现代文翻译任务的影响。


## Dependencies
bert4keras的安装可以参考[**bert4keras的安装过程**](https://github.com/bojone/bert4keras#%E4%BD%BF%E7%94%A8)，理论上有很多种tf+keras的组合：
<blockquote><strong>关于bert4keras的环境组合</strong>
  
- 支持tf+keras和tf+tf.keras，后者需要提前传入环境变量TF_KERAS=1。

- 当使用tf+keras时，建议2.2.4 <= keras <= 2.3.1，以及 1.14 <= tf <= 2.2，不能使用tf 2.3+。

- keras 2.4+可以用，但事实上keras 2.4.x基本上已经完全等价于tf.keras了，因此如果你要用keras 2.4+，倒不如直接用tf.keras。
</blockquote>

本文采用以下tf、keras组合：

- Python 3.7 
- tensorflow-gpu==1.15
- keras==2.3.1
- bert4keras==0.11.1

## Quick Start
### 环境配置准备
```
pip install tensorflow-gpu==1.15
pip install keras==2.3.1
pip install bert4keras==0.11.1
pip install h5py==2.10.0   # 如不安装，h5py版本问题会导致keras无法load模型权重。（本机的tf+keras环境有这个问题，可以先忽略这一行，有问题再安装。） 
```
除此之外，模型测试时计算BLEU分数需要用到`multi-bleu.perl`脚本，如果环境中没有perl，可以前往[官网](https://www.perl.org/get.html)安装

### 预训练模型下载
- **BERT-base-chinese**:  [chinese_L-12_H-768_A-12](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
- **RoBERTa-base-chinese**:  [chinese_roberta_wwm_ext_L-12_H-768_A-12](https://drive.google.com/open?id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94) 或去[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)下载
- **_Guwenbert_** : [Guwenbert的开源版本](https://github.com/Ethan-yt/guwenbert)是pytorch权重，因此需要转化为tf权重。
可以自行转换为tf格式，也可以使用我转换的版本：[百度网盘](https://pan.baidu.com/s/1heS4B3wZypJjKuhtpIF7Lg) 提取码：vcdp

### 模型参数修改
如果需要修改模型的训练、预测参数，请前往`config.py`中调整`batch size`、`MAX_LEN`、翻译方向（古至今还是今至古）、预训练模型参数位置等。

### 模型训练
```
python train.py
```

### 模型预测生成文本，并进行BLEU打分
```
python eval.py
```
