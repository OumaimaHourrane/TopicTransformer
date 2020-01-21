import random
import contants
import csv
import numpy as np
import pandas as pd

class corpus:
    def __init__(self,filename):
        self.word_indexes, self.vocabs, self.docs = self.load_doc(filename)
        self.num_docs = len(self.docs)
        self.num_words = len(self.word_indexes)

    def load_doc(self, filename):
        word_indexs = {contants.WORD_PAD: 0}
        vocabs = [contants.WORD_PAD]
        docs = []
        #file = open(filename,'r',encoding = "ISO-8859-1")
        file = pd.read_csv(filename, sep=',')
        file = file['text']
        for line in file:
            tokens = line.split(' ')
            doc = []
            for i in range(len(tokens)):
                word = tokens[i]
                if word in word_indexs:
                    doc.append(word_indexs[word])
                else:
                    index = len(word_indexs)
                    doc.append(index)
                    word_indexs[word] = index
                    vocabs.append(word)
            docs.append(doc)
        return word_indexs, vocabs, docs
    def index_batching(self, batch_size):
        indexes = [i for i in range(self.num_docs)]
        random.shuffle(indexes)
        num_batches = self.num_docs // batch_size
        batches = []
        for b in range(num_batches):
            batch_indexes = indexes[b*batch_size:(b+1)*batch_size]
            batches.append(batch_indexes)
        #last batch
        batch_indexes = indexes[-batch_size:]
        batches.append(batch_indexes)
        return  batches
