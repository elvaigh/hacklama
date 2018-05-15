import pandas as pd
import numpy as np
from os.path import basename
from scipy import spatial
def MWV(text,model):
	if not type(text) is str:return np.zeros(300)
	text=text.split()
	tmp=np.zeros(300)
	for i in text:
			try:tmp=np.add(tmp,model.wv[i])
			except:pass
	return tmp/len(text)
def ensText(text,model):
	text=text.split()
	tmp=[]
	for i in text:
			try:tmp+=[model.wv[i]]
			except:pass
	return tmp
def sim(a,b):return 1 - spatial.distance.cosine(a, b)
def meanSim(text,title):
	s=0
	for v in text:s+=sim(v,title)
	return s/(1+len(text))
from gensim.models.wrappers import FastText
model = FastText.load_fasttext_format('/home/celvaigh/these/divers/wiki.fr/wiki.fr.bin')
fr="wiki.fr.bin"
en="wiki.en.bin"
#model = word_vectors = KeyedVectors.load_word2vec_format('/home/celvaigh/these/divers/wiki.fr/wiki.fr.bin', binary=True)
def computeCorpusSims(name,lg):
	if lg=="fr":model = FastText.load_fasttext_format(fr)
	else:model = FastText.load_fasttext_format(en)
	data=pd.read_csv(name, sep='\t')
	texts=data["text"]
	titles=data["title"]
	size=len(texts)
	sims=[]
	for i in range(size):sims+=[meanSim(ensText(texts[i],model),MWV(titles[i],model))]
	sims.sort()
	return sims

def computeCorpusSims(name):
	data=pd.read_csv(name, sep='\t')
	texts=data["text"]
	if not "yt" in name:titles=data["title"]
	else:titles=data["video-title"]
	size=len(texts)
	sims=[]
	for i in range(size):sims+=[meanSim(ensText(texts[i],model),MWV(titles[i],model))]
	return sims
filenam='../test2/storyzy_en_test2.tsv'
sims=computeCorpusSims(filenam)
a,b=basename(filenam).split(".")
filenam="../"+a+"_sim_texte_titre.txt"
with open(filenam,"w+") as f:
	f.write("sim_texte_tittre\n")
	for i in sims:f.write("%s\n" % i)
"""filenam='../test2/storyzy_fr_test2.tsv'
sims=computeCorpusSims(filenam)
a,b=basename(filenam).split(".")
filenam="../"+a+"_sim_texte_titre.txt"
with open(filenam,"w+") as f:
	f.write("sim_texte_tittre\n")
	for i in sims:f.write("%s\n" % i)
filenam='../test2/storyzy_yt_test2.tsv'
sims=computeCorpusSims(filenam)
a,b=basename(filenam).split(".")
filenam="../"+a+"_sim_texte_titre.txt"
with open(filenam,"w+") as f:
	f.write("sim_texte_tittre\n")
	for i in sims:f.write("%s\n" % i)"""
