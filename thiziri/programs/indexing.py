import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pandas as pd
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import ntpath

"""
It returns the file name extracted from a path
"""
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class Index:
    """ Inverted index datastructure """

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)

        return [self.documents.get(id, None) for id in self.index.get(word)]

    def add(self, document, doc_id):
        """
        Add a document string to the index
        """

        self.__unique_id = doc_id

        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)

        self.documents[self.__unique_id] = document          


index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

#corpus_train = ["/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_en_train.tsv",
#    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_fr_train.tsv",
#    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/train/storyzy_yt_train.tsv"]

corpus = ["/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_en_test2.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_fr_test2.tsv",
    "/home/thiziri/Documents/DOCTORAT/EVENTS/HACKATON_CORIA18/test2/storyzy_yt_test2.tsv"]
documents = []
for doc in corpus:
    df =  pd.read_csv(doc, header=0, delimiter="\t")
    #print (df1)
    title_column = 'title' if "yt" not in doc else "video-title"
    id_column = 'id'
    for _,row in tqdm(df.iterrows()):
        txt = str(row[title_column]) + ' ' + str(row['text'])
        #print(txt)
        id_doc = row[id_column]
        index.add(txt, id_doc)
        documents.append(txt.strip().split())
        #break 

    dct = Dictionary(documents)  # fit dictionary
    #print(dct)

    print("Index ok.")
    #print(index.documents)

    print("tfidf training...")
    tfidf = {}

    print("file: ", path_leaf(doc))
    df =  pd.read_csv(doc, header=0, delimiter="\t")
    all_text = []
    i = 0
    for _,row in df.iterrows():
        all_text.append(str(row[title_column]))
        all_text.append(str(row['text']))
        id_t = str(row[id_column])
        tfidf[id_t] = i
        i +=1
    corpus_txt = [dct.doc2bow(txt.split()) for txt in all_text]
    model = TfidfModel(corpus_txt)
    #print(corpus_txt[0])


    out = open("sim_tfidf_"+path_leaf(doc).split(".")[0]+".tsv", "w")
    print("Ccomputing similarities ...")
    for d in tfidf:
        title = corpus_txt[tfidf[d]]
        title_words = [e[0] for e in title]
        text = corpus_txt[tfidf[d]+1]
        sim = 0.0
        intersect = [e for e in text if e[0] in title_words]
        sim = sum([e[1] for e in intersect])
        #print(d, sim)
        out.write("{id}\t{sim}\n".format(id=d, sim=sim))

