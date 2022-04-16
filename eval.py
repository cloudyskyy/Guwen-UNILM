import numpy as np
from bert4keras.models import build_transformer_model
from config import ROBERTA_DICT_PATH, ROBERTA_CHECKPOINT_PATH, ROBERTA_CONFIG_PATH,BERT_CONFIG_PATH, BERT_CHECKPOINT_PATH, BERT_DICT_PATH, BEST_MODEL_PATH, PREDICT_PATH, ANCIENT_TEST_PATH, \
	MODERN_TEST_PATH, GUWEN_DICT_PATH, GUWEN_CHECKPOINT_PATH, GUWEN_CONFIG_PATH, modern2ancient
from bert4keras.tokenizers import Tokenizer, load_vocab
from train import AutoTitle
from train import Ancient2Modern, use_guwenbert
import subprocess


def build_model():
	config_path = GUWEN_CONFIG_PATH if use_guwenbert else ROBERTA_CONFIG_PATH
	checkpoint_path = GUWEN_CHECKPOINT_PATH if use_guwenbert else ROBERTA_CHECKPOINT_PATH
	dict_path = GUWEN_DICT_PATH if use_guwenbert else ROBERTA_DICT_PATH

	token_dict, keep_tokens = load_vocab(
		dict_path=dict_path,
		simplified=True,
		startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
	)
	tokenizer = Tokenizer(token_dict, do_lower_case=True)

	model = build_transformer_model(
		config_path,
		checkpoint_path,
		application='unilm',
		# keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
	)

	# 加载训练好的模型
	model.load_weights(BEST_MODEL_PATH)

	autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=50)

	text = '却话巴山夜雨时'
	token_ids, segment_ids = tokenizer.encode(text)
	inputs = np.array([token_ids, segment_ids])
	inputs = [np.array([i]) for i in inputs]
	print(autotitle.predict(inputs, np.empty((1, 0), dtype=int), states=None))
	print(autotitle.generate("却话巴山夜雨时"))
	return autotitle


def execute(li):
	(status, output) = subprocess.getstatusoutput(li)
	print(status, output)
	return output



def sep(str):
	s = str.strip('\n')
	l = []
	for c in s:
		l.append(c)
	new_str = ' '.join(l)
	return new_str


def predict(autotitle, modern2ancient=modern2ancient, use_guwenbert=use_guwenbert):
	fp = [MODERN_TEST_PATH, ANCIENT_TEST_PATH]
	if use_guwenbert:
		md='G'
	else:
		md='R'
	if modern2ancient:
		flag = [0, 1]
		direction = 'm2a'
	else:
		flag = [1, 0]
		direction = 'a2m'

	save_path = PREDICT_PATH

	f = open(fp[flag[0]], 'r', encoding='utf-8')
	out = open(save_path, 'w', encoding='utf-8')
	gold = open(fp[flag[1]], 'r', encoding='utf-8').readlines()
	lines = f.readlines()


	i = 0
	for l in lines:
		print('{}/{}'.format(i + 1, len(lines)))
		generated = autotitle.generate(l.strip('\n'))
		out.write(sep(generated) + '\n')
		print('原句:', l.strip('\n'))
		print('参考翻译:', gold[i].strip('\n'))
		print(u'翻译结果:', blank(generated))
		print('\n')
		i += 1

	return save_path


def blank(s):
	new_s = []
	for c in s.strip('\n'):
		new_s.append(c)
	return ' '.join(new_s)

def bleu(path):
	fp = [MODERN_TEST_PATH, ANCIENT_TEST_PATH]
	if modern2ancient:
		flag = 1
	else:
		flag = 0
	string = 'perl multi-bleu.perl {} < {}'.format(fp[flag], path)
	print(string)
	texts = execute(string)
	f = open(path+'_bleu.txt', 'w', encoding='utf-8')
	f.writelines(texts)
	f.close()


def main():
	autotitle = build_model()
	path = predict(autotitle, modern2ancient=modern2ancient)
	bleu(path)
	'''while True:
		s = input()
		print(autotitle.generate(s))'''


if __name__ == '__main__':
	main()

