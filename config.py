# -*- coding: utf-8 -*-

MAX_LEN = 128
batch_size = 30
steps_per_epoch = 1000
epochs = 15  # 这里的epochs指的是训练多少轮的bs*step，不是指数据集的数据量，当然可以调整bs、step使bs*step=len(数据集)

Ancient2Modern = False
use_guwenbert = True

modern2ancient = not Ancient2Modern

# 数据集设置
dataset = 'hist'
MODERN_TRAIN_PATH = f'data/{dataset}/modern-train.txt'
ANCIENT_TRAIN_PATH = f'data/{dataset}/ancient-train.txt'
MODERN_TEST_PATH = f'data/{dataset}/modern-test.txt'
ANCIENT_TEST_PATH = f'data/{dataset}/ancient-test.txt'

# BERT模型权重位置 需要下载预训练权重后自行修改路径
BERT_CONFIG_PATH = r'pretrained_models\chinese_L-12_H-768_A-12\bert_config.json'
BERT_CHECKPOINT_PATH = r'pretrained_models\chinese_L-12_H-768_A-12\bert_model.ckpt'
BERT_DICT_PATH = r'pretrained_models\chinese_L-12_H-768_A-12\vocab.txt'

# RoBERTa模型权重位置 需要下载预训练权重后自行修改路径
ROBERTA_CONFIG_PATH = r'pretrained_models\chinese_roberta_wwm_ext_L-12_H-768_A-12\bert_config.json'
ROBERTA_CHECKPOINT_PATH = r'pretrained_models\chinese_roberta_wwm_ext_L-12_H-768_A-12\bert_model.ckpt'
ROBERTA_DICT_PATH = r'pretrained_models\chinese_roberta_wwm_ext_L-12_H-768_A-12\vocab.txt'

# GuwenBERT模型权重位置 需要下载预训练权重后自行修改路径
GUWEN_CONFIG_PATH = r'pretrained_models\guwenbert-base-tf\config.json'
GUWEN_CHECKPOINT_PATH = r'pretrained_models\guwenbert-base-tf\roberta.ckpt'
GUWEN_DICT_PATH = r'pretrained_models\guwenbert-base-tf\vocab.txt'

# 模型预测文件位置
PREDICT_PATH = f'outputs/{dataset}/predict.txt'

DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
BEST_MODEL_PATH = 'best_model.weights'
