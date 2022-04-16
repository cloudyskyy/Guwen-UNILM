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
- **_Guwenbert_** : [Guwenbert的开源版本](https://github.com/Ethan-yt/guwenbert)是pytorch权重，而本文使用的bert4keras是基于TensorFlow+keras框架，因此需要将pytorch权重转化为tf权重。可以自行转换为tf格式，也可以使用我转换的版本：[百度网盘](https://pan.baidu.com/s/1heS4B3wZypJjKuhtpIF7Lg) 提取码：vcdp

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
## 模型输出样例（以现代文 → 古文为例）
```
原句: 太 宰 官 内 心 惭 愧 ， 要 求 国 书 。
参考翻译: 太 宰 官 愧 服 ， 求 国 书 。
翻译结果: 太 史 内 愧 ， 求 国 书 。
```

```
原句: 日 本 知 不 能 使 良 弼 屈 服 ， 派 使 引 十 二 人 去 见 日 本 国 王 ， 而 将 良 弼 送 至 对 马 岛 。
参考翻译: 日 本 知 不 可 屈 ， 遣 使 介 十 二 人 入 觐 ， 仍 遣 人 送 良 弼 至 对 马 岛 。
翻译结果: 日 本 知 不 能 屈 ， 遣 使 引 十 二 人 见 日 本 国 主 ， 送 弼 于 对 马 岛 。
```

```
原句: 丞 相 伯 颜 伐 宋 ， 良 弼 建 议 ： 宋 重 兵 在 扬 州 ， 最 好 以 大 军 先 捣 钱 塘 。
参考翻译: 丞 相 伯 颜 伐 宋 ， 良 弼 言 ： 宋 重 兵 在 扬 州 ， 宜 以 大 军 先 捣 钱 唐 。
翻译结果: 丞 相 伯 颜 伐 宋 ， 良 弼 言 ： 宋 重 兵 在 扬 州 ， 宜 以 大 兵 先 捣 钱 塘 。
```

## Discussion and Extension
- 事实上，本人在20年末提出的数据集已经不够先进了，20K-30K左右的数据量实在是比较小。小牛翻译（东北大学NLP组）在2022年2月开源的文言文-现代文平行语料，一共有967257个句对，地址在[文言文（古文）-现代文平行语料](https://github.com/NiuTrans/Classical-Modern)，与本文构建的数据集如出一辙，都为句子级语料，且规模在1000k左右，是我们的数据集规模的30-50倍。该数据集也可以直接使用我们Guwen-UNILM的代码运行。如有兴趣可以拿本仓库的代码去跑一下。除此之外，本文的UNILM模型与后来者BART、T5等作比较，可能性能也不占优势。后面由于本人升学等原因，研究方向大概率不会与古文翻译有关了。事实上这是一个有趣的课题，其实也有不少点能做。
