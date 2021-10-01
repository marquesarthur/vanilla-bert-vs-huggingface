import os
import time

from gensim.test.utils import datapath
from gensim.models.fasttext import FastText, save_facebook_model, load_facebook_model
from gensim.utils import tokenize
from gensim import utils
from gensim.models.callbacks import CallbackAny2Vec

# hyperparameters
window = 5
min_count = 2
vector_size = 100
epoch = 10

# input paths
input_corpus = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'corpus.txt')
model_bin = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SO_fasttext_vectors_200.bin")


class MyIter:

    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        # path = datapath()
        with utils.open(self.input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))


# https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
# https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
class LossCallback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.start = time.time()

    def on_epoch_end(self, model):
        # loss = model.get_latest_training_loss()
        end = time.time()
        hours, rem = divmod(end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        # print('Loss after epoch {}: {:.8f}'.format(self.epoch, loss))
        print("[{}] Time elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(self.epoch), int(hours), int(minutes), seconds))
        self.epoch += 1
        self.start = time.time()


# https://stackoverflow.com/questions/67573416/unable-to-recreate-gensim-docs-for-training-fasttext-typeerror-either-one-of-c
# these are the same parameters as R. Silva + computing training loss for logging purposes
model = FastText(vector_size=vector_size, min_n=2, max_n=5, epochs=epoch)
print('build vocab...')
start = time.time()
model.build_vocab(MyIter(input_corpus))
total_examples = model.corpus_count
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("Time elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

print(model.corpus_count)
print('\n\ntrain model...')
model.train(MyIter(input_corpus), total_examples=total_examples, epochs=epoch, callbacks=[LossCallback()])

print(model.wv[' '])
print(model.wv['use'])
print(model.wv['even'])

print('\n\nsaving model...')
start = time.time()
save_facebook_model(model, model_bin)
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("Time elapsed {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

# load model?
# print('loading saved model...')
# fb_model = load_facebook_model(model_bin)
# print(fb_model.wv[' '])
# print(fb_model.wv['use'])
# print(fb_model.wv['thisIsaReallyLongN'])
