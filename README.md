# Guwen-UNILM: Machine Translation Between Ancient and Modern Chinese Based on Pre-Trained Models
<div  align="center">   
 <img src="https://user-images.githubusercontent.com/30574139/163604086-13213a39-ab35-42d1-806f-04f48914b6f3.png" width = "600" height = "400" alt="图片名称" align=center />
</div> 

本仓库是基于[bert4keras](https://github.com/bojone/bert4keras)实现的古文-现代文翻译模型。具体使用了基于掩码自注意力机制的[UNILM(Li al., 2019)](https://arxiv.org/abs/1905.03197)预训练模型作为翻译系统的backbone。我们首先使用了普通的中文（现代文）BERT、Roberta权重作为UNILM的初始权重以训练UNILM模型（具体在文中分别为B-UNILM以及R-UNILM）。为了更好的使UNILM模型适应古文的特性，我们尝试使用了在古文预训练模型[Guwen-BERT](https://github.com/Ethan-yt/guwenbert)，作为UNILM的初始权重，并且获得了最优的效果。除此之外，我们研究了分词对于古文-现代文翻译任务的影响。


## Dependencies
- Python 3.7 
- tensorflow-gpu==1.15
- keras==2.3.1
- bert4keras==0.11.1
## Quick Start
bert4keras的安装可以参考[**bert4keras的安装过程**](https://github.com/bojone/bert4keras#%E4%BD%BF%E7%94%A8)。
<blockquote><strong>关于bert4keras的环境组合</strong>
  
- 支持tf+keras和tf+tf.keras，后者需要提前传入环境变量TF_KERAS=1。

- 当使用tf+keras时，建议2.2.4 <= keras <= 2.3.1，以及 1.14 <= tf <= 2.2，不能使用tf 2.3+。

- keras 2.4+可以用，但事实上keras 2.4.x基本上已经完全等价于tf.keras了，因此如果你要用keras 2.4+，倒不如直接用tf.keras。
</blockquote>

```
pip install tensorflow-gpu==1.15
pip install keras==2.3.1
pip install bert4keras==0.11.1
```

