import nltk
from nltk import word_tokenize
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pandas as pd
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import ntpath

from indexing import path_leaf , Index

index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

corpus_train = ["/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_en_train.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_fr_train.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_yt_train.tsv"]

corpus_test1 = ["/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test1/storyzy_en_test1.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test1/storyzy_fr_test1.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test1/storyzy_yt_test1.tsv"]

corpus_test2 = ["/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_en_test2.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_fr_test2.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_yt_test2.tsv"]

documents = {}
for doc in corpus_test1:
    print(doc)
    out = open("num_superlatives"+path_leaf(doc).split(".")[0]+".tsv", "w")
    out.write("doc_id\tsup_num_title\tsup_num_text\n")
    df =  pd.read_csv(doc, header=0, delimiter="\t")
    #print (df1)
    title_column = 'title' if "yt" not in doc else "video-title"
    id_column = 'id'
    for _,row in tqdm(df.iterrows()):
        text = str(row['text'])
        title = str(row[title_column])
        txt = ' '.join([title, text])
        #print(txt)
        id_doc = row[id_column]
        index.add(txt, id_doc)
        #documents.append(txt.strip().split())
        title_tags = []
        text_tags = []
        if 'en' in doc:
            title_token = word_tokenize(title)
            title_tags = nltk.pos_tag(title_token)
            text_token = word_tokenize(text)
            text_tags = nltk.pos_tag(text_token)
            #print("\t".join([str(id_doc), str(len([t for t in title_tags if t[1]=="JJS"])), str(len([t for t in text_tags if t[1]=="JJS"]))]))
            out.write("\t".join([str(id_doc), str(len([t for t in title_tags if t[1]=="JJS"])), str(len([t for t in text_tags if t[1]=="JJS"]))])+"\n")
        if 'fr' in doc or 'yt' in doc:
            title_num_superlatives = 0
            text_num_superlatives = 0
            if 'le plus' in title.lower():
                sub_strings = title.lower().split("le plus")
                title_num_superlatives = len([t.split()[0] for t in sub_strings if len(t.split())>0])
            if 'le plus' in text.lower():
                sub_strings = text.lower().split("le plus")
                text_num_superlatives = len([t.split()[0] for t in sub_strings if len(t.split())>0])
            out.write("\t".join([str(id_doc), str(title_num_superlatives), str(text_num_superlatives)])+"\n")
print("Finished.")
