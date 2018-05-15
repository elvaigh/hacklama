import collections, nltk
from os.path import basename
# we first tokenize the text corpus
import pandas as pd
filenam="../test2/storyzy_en_test2.tsv"
def buildCorpus(name):
		data=pd.read_csv(name, sep='\t')
		corpus=""
		texts=data["text"]
		#titles=data["title"]
		for text in texts:corpus+=text
		return corpus,texts
		
a,b=basename(filenam).split(".")

corpus,texts=buildCorpus(filenam)
tokens = nltk.word_tokenize(corpus)
def unigram(tokens):    
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model [f] = 1
            continue
    for word in model:
        model[word] = model[word]/float(sum(model.values()))
    return model
#computes perplexity of the unigram model on a testset  
def perplexity(testset, model):
    testset = testset.split()
    
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(max(N,1))) 
    return perplexity
testset1 = "Monty"
testset2 = "abracadabra gobbledygook rubbish"

model = unigram(tokens)
perpx=[]
for text in texts:perpx+=[perplexity(text,model)]

filenam="../"+a+"_perplexity.txt"
with open(filenam,"w+") as f:
	for i in perpx:f.write("%s\n" % i)
def classifier(name,checked,min_e):
	data=pd.read_csv(name, sep='\t')
	typ=data["type"]
	size=len(typ)
	typ2=[1 for i in range(size)]
	s=0
	print(len(typ)==len(typ2))
	for i in range(size):
		if checked[i]>min_e:typ2[i]=0
	for i in range(size):
		if (typ[i]=="trusted" and typ2[i]==1) or (typ[i]!="trusted" and typ2[i]==0):s+=1
	return s/size
	
#print(classifier(filenam,perpx,5000))
