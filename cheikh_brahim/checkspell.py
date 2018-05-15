import enchant
import string
import pandas as pd
from os.path import basename
import re
from nltk import word_tokenize
def textCheck(name):
	data=pd.read_csv(name, sep='\t')
	d = enchant.Dict("fr_FR")
	texts=data["text"]
	ss=[]
	for text in texts:
		s=0
		tokens=word_tokenize(text)
		
		for t in tokens:
			if (d.check(t) is False and re.match('^[a-zA-Z ]*$',t)) or "vaccin" in t:s+=1
		#print(100*s/len(tokens))
		#exit()
		#if s==0:print("zero   " ,text);exit()
		ss+=[100*s/(len(tokens)+1)]
	
	return ss

filenam="../test2/storyzy_yt_test2.tsv"
checked=textCheck(filenam)
a,b=basename(filenam).split(".")
filenam="../checkspelling/"+a+"_checkspelling.txt"
with open(filenam,"w+") as f:
	f.write("check_spelling\n")
	for i in checked:f.write("%s\n" % i)
#min 4 and max 783
def classifier(name,checked,min_e):
	data=pd.read_csv(name, sep='\t')
	typ=data["type"]
	size=len(typ)
	typ2=[1 for i in range(size)]
	s=0
	for i in range(size):
		if checked[i]>min_e:typ2[i]=0
		if (typ[i]=="trusted" and typ2[i]==1) or (typ[i]!="trusted" and typ2[i]==0):s+=1
	return s/size
#ev=classifier('hackathon-train/train/storyzy_en_train.tsv',checked,100)
#print(ev)
