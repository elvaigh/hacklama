# generate for each text (in one line) the five topic words with the highest proba 

import os, sys, re
import nltk, random, spacy 
import pickle, gensim 
# nltk.download('wordnet')
# nltk.download('stopwords')
# from nltk.corpus import wordnet as wn
# from nltk.stem.wordnet import WordNetLemmatizer
spacy.load('en')
from spacy.en import English
from gensim import corpora
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
import numpy as np

parser = English()
en_stop = set(nltk.corpus.stopwords.words('english'))
black_list = ['@card@']

# using already tokenized text 
def tokenize(text):
	lda_tokens = []
	tokens = parser(text)
	for token in tokens:
		if token.orth_.isspace():
			continue
		# elif token.like_url:
		# 	lda_tokens.append('URL')
		# elif token.orth_.startswith('@'):
		# 	lda_tokens.append('SCREEN_NAME')
		else:
			lda_tokens.append(token.lower_)
	return lda_tokens

# def get_lemma(word):
# 	lemma = wn.morphy(word)
# 	if lemma is None:
# 		return word
# 	else:
# 		return lemma
	
def prepare_text_for_lda(text):
	tokens = tokenize(text)
	tokens = [token for token in tokens if len(token) > 2]
	tokens = [token for token in tokens if token not in en_stop]
	tokens = [token for token in tokens if token not in black_list]
	# tokens = [get_lemma(token) for token in tokens]
	return tokens

# fr="fr.vec"   # 100 dimension 
# en="en.vec"

# en_model = KeyedVectors.load_word2vec_format(en)
# fr_model = KeyedVectors.load_word2vec_format(fr)

with open("storyzy_en_train.tsv.normalized.tsv.text.lemma.txt") as FT:
	next(FT)
	for line in FT:
		text_data = []
		line = line.strip('\n')
		tokens = prepare_text_for_lda(line)
		text_data.append(tokens)

		dictionary = corpora.Dictionary(text_data)
		corpus = [dictionary.doc2bow(text) for text in text_data]
		pickle.dump(corpus, open('corpus.pkl', 'wb'))
		dictionary.save('dictionary.gensim')	

		NUM_TOPICS = 5
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
		ldamodel.save('model5.gensim')
		topics = ldamodel.print_topics(num_words=4)

		topics_dic = {}
		for topic in topics:
			proba = []
			word = []

			m = re.search(r'\(\d+\, (.*?) \+ (.*?) \+ (.*?) \+ (.*?)\'.*', str(topic))

			t1 = m.group(1).replace('"','').lstrip('\'')
			t2 = m.group(2).replace('"','')
			t3 = m.group(3).replace('"','')
			t4 = m.group(4).replace('"','')

			proba1 = t1.split('*')[0]
			w1 = t1.split('*')[1]

			proba2 = t2.split('*')[0]
			w2 = t2.split('*')[1]

			proba3 = t3.split('*')[0]
			w3 = t3.split('*')[1]

			proba4 = t4.split('*')[0]
			w4 = t4.split('*')[1]

			proba.extend((proba1, proba2, proba3, proba4))
			word.extend((w1, w2, w3, w4)) 

			for (x, y) in zip(word, proba):
				if x not in topics_dic:
					topics_dic[x] = y 
				else:
					if y > topics_dic[x]:
						topics_dic[x] = y

		sorted_keys = sorted(topics_dic, key=topics_dic.get, reverse=True)
		result_topic = []
		for r in sorted_keys[0:5]:
			# topics_dic[r]: probability 
			result_topic.append(r)

		# print(result_topic)
		tmp = np.zeros(300)  # for english 
		for word in result_topic:
			# print(word)
			embedding = 
			try:
				tmp = np.add(tmp, embedding)
			except:
				pass

		average = tmp/5 






