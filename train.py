#! -*- coding: utf-8 -*-
# bert/roberta/GuwenBert做Seq2Seq任务，采用UNILM方案

from __future__ import print_function
import glob
import pickle
import subprocess
import time
from builtins import open as myopen
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from config import ROBERTA_DICT_PATH, ROBERTA_CHECKPOINT_PATH, ROBERTA_CONFIG_PATH, BERT_CONFIG_PATH,BERT_CHECKPOINT_PATH, BERT_DICT_PATH,\
    ANCIENT_TEST_PATH, MODERN_TEST_PATH, PREDICT_PATH, ANCIENT_TRAIN_PATH, MODERN_TRAIN_PATH, MAX_LEN,\
    GUWEN_CHECKPOINT_PATH, GUWEN_CONFIG_PATH, GUWEN_DICT_PATH, BEST_MODEL_PATH, Ancient2Modern, use_guwenbert,batch_size,steps_per_epoch,\
    epochs
import config
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K

# bert配置
config_path = GUWEN_CONFIG_PATH if use_guwenbert else ROBERTA_CONFIG_PATH
checkpoint_path = GUWEN_CHECKPOINT_PATH if use_guwenbert else ROBERTA_CHECKPOINT_PATH
dict_path = GUWEN_DICT_PATH if use_guwenbert else ROBERTA_DICT_PATH


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)
ancients = []
moderns = []

a_path = ANCIENT_TRAIN_PATH
m_path = MODERN_TRAIN_PATH

# print(a_path,m_path)
for line in open(a_path, 'r',encoding='utf-8'):
    # print(line)  # 对每一行进行操作即可
    ancients.append(line.strip('\n'))
for line in open(m_path, 'r',encoding='utf-8'):
    # print(line)  # 对每一行进行操作即可
    moderns.append(line.strip('\n'))
para = []

print('Ancient2Modern: ', Ancient2Modern)
if Ancient2Modern:
    for i in range(len(ancients)):
        para.append(ancients[i] + '\n' + moderns[i])
else:
    for i in range(len(ancients)):
        para.append(moderns[i] + '\n' + ancients[i])
print(f'训练数据大小{len(para)}\n样例:', para[0].replace('\n', '  |  '))


def execute(li):
    (status, output) = subprocess.getstatusoutput(li)
    print(status, output)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, txt in self.sample(random):
            text = txt
            text = text.split('\n')
            if len(text) == 2:
                src = text[0]
                tgt = text[1]
                token_ids, segment_ids = tokenizer.encode(
                    src, tgt, maxlen=MAX_LEN
                )
                #print(token_ids,segment_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        K.print_tensor(y_true, message='y_true=')
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)
model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        # print(token_ids, type(token_ids), token_ids.shape)
        # print(output_ids, type(output_ids), output_ids.shape)
        token_ids = np.concatenate([token_ids, output_ids], 1)
        # print(token_ids)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=4):
        max_c_len = MAX_LEN - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        # print(type(token_ids))
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


def just_show(epoch):
    s1 = u'不知乘月几人归'
    s2 = u'被吞没了多时的满月一下子跳了出来像一个刚出炼炉的金盘'
    print(u'生成诗句:', autotitle.generate(s1))
    if epoch % 5 == 0 and epoch > 30:
        print('epoch:' + str(epoch))


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e105

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        print(logs)
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(BEST_MODEL_PATH)
        # just_show(epoch)
        loss_on_epoch_end.append(logs['loss'])


if __name__ == '__main__':
    loss_on_epoch_end = []
    evaluator = Evaluator()
    train_generator = data_generator(para, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )
    loss_on_epoch_end = np.array(loss_on_epoch_end)
    # f = myopen('Roberta-UNILM_a2m.loss', 'wb')
    # pickle.dump(loss_on_epoch_end, f)

else:
    model.load_weights(BEST_MODEL_PATH)
