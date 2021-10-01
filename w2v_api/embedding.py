# the following codes aree packaged as a py file that can be import in other python scripts.
# this user own package is named as "data_prepare"


import logging
import math
import time
import os

import gensim.downloader as api
import numpy as np
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText, save_facebook_model, load_facebook_model
# from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim import corpora
from scipy.spatial import distance


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html


def ConvertVectorSetToVecAverageBased(vectorSet, ignore=None):
    if ignore is None:
        ignore = []
    if len(ignore) == 0:
        return np.mean(vectorSet, axis=0)
    else:
        return np.dot(np.transpose(vectorSet), ignore) / sum(ignore)


def PhraseToVec(phrase, model=None, fasttext=False):
    if not model:
        raise Exception("No pre-trained model defined")
    cachedStopWords = stopwords.words("english")
    phrase = phrase.lower()
    wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
    vectorSet = []
    for aWord in wordsInPhrase:
        try:
            if not fasttext:
                wordVector = model[aWord]
            else:
                wordVector = model.wv[aWord]
            vectorSet.append(wordVector)
        except:
            pass
    if not fasttext:
        return ConvertVectorSetToVecAverageBased(vectorSet)
    else:
        return np.mean(vectorSet, axis=0)


# https://bitbucket.org/yunazzang/aiwiththebest_byor/src/master/PhraseSimilarity.py
class PhraseVector(object):
    def __init__(self, phrase, model=None, fasttext=False):
        self.vector = PhraseToVec(phrase, model=model, fasttext=fasttext)

    def similarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (
                np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity


class Word2Vec(object):

    def __init__(self):
        self.model = None

    def similarity(self, x, y):
        """
        Computes the cosine similarity of two sentences.
        First, each sentence is converted into its normalized length vector representation.
        Then, the cosine similarity between sentences vectors are computed.
        Uses sklearn cosine similarity function, which works with both dense and sparse vectors.

        For more details, see:
            https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        """
        if not self.model:
            raise Exception("You cannot use a similarity function without a trained model")

        x_vector = PhraseVector(x, model=self.model)
        y_vector = PhraseVector(y, model=self.model)

        result = x_vector.similarity(y_vector.vector)
        return result

    def vector(self, x):
        x_vector = PhraseVector(x, model=self.model)
        return x_vector.vector

    def load(self, gensim_pre_trained_model="word2vec"):
        start = time.time()
        logging.info("Loading model")
        self.model = api.load(gensim_pre_trained_model)  # model = api.load("word2vec-google-news-300")
        end = time.time()
        logging.info(">> model loaded")
        logging.info(">> %s" % (end - start))


class SOWord2Vec(object):

    def __init__(self):
        self.model = None

    def similarity(self, x, y):
        """
        Computes the cosine similarity of two sentences.
        First, each sentence is converted into its normalized length vector representation.
        Then, the cosine similarity between sentences vectors are computed.
        Uses sklearn cosine similarity function, which works with both dense and sparse vectors.

        For more details, see:
            https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        """
        if not self.model:
            raise Exception("You cannot use a similarity function without a trained model")

        # x_vector = PhraseVector(x, model=self.model)
        # y_vector = PhraseVector(y, model=self.model)
        #
        # result = x_vector.similarity(y_vector.vector)
        # return result
        stop_words = stopwords.words("english")
        sentence_x = [w for w in x.lower().split() if w not in stop_words]
        sentence_y = [w for w in y.lower().split() if w not in stop_words]

        x_vec = self.avg_feature_vector(sentence_x, self.model, self.model.vector_size)
        y_vec = self.avg_feature_vector(sentence_y, self.model, self.model.vector_size)

        try:
            cosine_similarity = np.inner(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
            if math.isnan(cosine_similarity):
                cosine_similarity = 0.
        except:
            cosine_similarity = 0.
        return cosine_similarity

    def avg_feature_vector(self, sentence, model, num_features):
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in sentence:
            if word in self.model:
                feature_vec = np.add(feature_vec, model[word])
                n_words += 1
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def vector(self, x):
        x_vector = PhraseVector(x, model=self.model)
        return x_vector.vector

    def load(self, file_name="SO_vectors_200.bin"):
        start = time.time()
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
        logging.info(f"Loading model from {file_path}")
        self.model = KeyedVectors.load_word2vec_format(file_path, binary=True)
        end = time.time()
        logging.info(">> model loaded")
        logging.info(">> %s" % (end - start))


class SOFasttext(object):

    def __init__(self):
        self.model = None

    def similarity(self, x, y):
        """
        Computes the cosine similarity of two sentences.
        First, each sentence is converted into its normalized length vector representation.
        Then, the cosine similarity between sentences vectors are computed.
        Uses sklearn cosine similarity function, which works with both dense and sparse vectors.

        For more details, see:
            https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
            https://github.com/RaRe-Technologies/gensim/blob/release-3.8.3/docs/notebooks/soft_cosine_tutorial.ipynb
        """
        if not self.model:
            raise Exception("You cannot use a similarity function without a trained model")

        stop_words = stopwords.words("english")

        sentence_x = [w for w in x.lower().split() if w not in stop_words]
        sentence_y = [w for w in y.lower().split() if w not in stop_words]

        x_vec = self.avg_feature_vector(sentence_x, self.model, self.model.vector_size)
        y_vec = self.avg_feature_vector(sentence_y, self.model, self.model.vector_size)

        try:
            cosine_similarity = np.inner(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
            if math.isnan(cosine_similarity):
                cosine_similarity = 0.
        except:
            cosine_similarity = 0.
        return cosine_similarity

    def avg_feature_vector(self, sentence, model, num_features):

        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in sentence:
            feature_vec = np.add(feature_vec, model.wv[word])
            n_words += 1
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def vector(self, x):
        x_vector = PhraseVector(x, model=self.model, fasttext=True)
        return x_vector.vector

    def load(self, file_name="SO_fasttext_vectors_200.bin"):
        start = time.time()
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
        logging.info(f"Loading model from {file_path}")
        self.model = load_facebook_model(file_path)
        end = time.time()
        logging.info(">> model loaded")
        logging.info(">> %s" % (end - start))


#

# print(w2v.similarity(x, y))
#
#
# x = "sorry, these are completely unrelated strings"
# y = "nothing"
# print(w2v.similarity(x, y))

#
# x = "Have you ever tried to used the method string.split(\"hello world\")"
# y = "try using the string.split method"
# sentence_obama = 'Obama speaks to the media in Illinois'
# sentence_president = 'The president greets the press in Chicago'
# sentence_orange = 'Having a tough time finding an orange juice press machine?'
#
# w2v = SOWord2Vec()
# w2v.load(file_name="SO_vectors_200.bin")
#
# # w2v = SOFasttext()
# # w2v.load(file_name="SO_fasttext_vectors_200.bin")
# print(w2v.similarity(x, y))
# print(w2v.similarity(sentence_obama, sentence_president))
# print(w2v.similarity(sentence_obama, sentence_orange))

# using np inner_dot
# SOWord2Vec()
#       0.8183837
#       0.40002513
#       0.1951475
# SOFasttext --- full
#       0.6530203
#       0.4777708
#       0.37745786
# SOFasttext --- small toy example
#       0.97848946
#       0.9238482
#       0.9784081

#
# using n_similarity <----- I think that this is more reasonable and similar to
# 0.9784895
# 0.9238482
# 0.97840816
#
