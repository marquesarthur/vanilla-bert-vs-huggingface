# https://radimrehurek.com/gensim/models/fasttext.html
# https://stackoverflow.com/questions/58876630/how-to-export-a-fasttext-model-created-by-gensim-to-a-binary-file
# from gensim.models.fasttext import FastText
# from gensim.test.utils import datapath
# from gensim.utils import tokenize
# from gensim import utils

# class MyIter(object):
#     def __iter__(self):
#         path = datapath('crime-and-punishment.txt')
#         with utils.open(path, 'r', encoding='utf-8') as fin:
#             for line in fin:
#                 yield list(tokenize(line))
#
#
# model4 = FastText(vector_size=100)
# model4.build_vocab(sentences=MyIter())
# total_examples = model4.corpus_count
# model4.train(sentences=MyIter(), total_examples=total_examples, epochs=5)
# sentences = [[" ", "Yes", "Who"], ["I", "Yes", "Chinese"]]
# model = FastText(sentences, size=4, window=3, min_count=1, iter=10, min_n=3, max_n=6, word_ngrams=0)
# model[' ']  # The way the word vector is obtained
# model.wv['you']  # The way the word vector is obtained

#
# import postgresql
# db = postgresql.open('pq://w2v:password123@127.0.0.1:5432/crokage')
#
# get_table = db.prepare("SELECT processedtitle, processedbody from postmin")
#
# # Streaming, in a transaction.
# with db.xact():
# 	for x in get_table.rows("tables"):
# 		print(x)
#
#
#
#
# Connection.query.load_chunks(collections.abc.Iterable(collections.abc.Iterable(parameters)))


# https://rizwanbutt314.medium.com/efficient-way-to-read-large-postgresql-table-with-python-934d3edfdcc
import psycopg2
from datetime import datetime

start = datetime.now()

connection = psycopg2.connect(
    dbname='crokage',
    user='w2v',
    password='password123',
    host='127.0.0.1',
    port=5432
)

# https://stackoverflow.com/questions/49266939/time-performance-in-generating-very-large-text-file-in-python
# https://rizwanbutt314.medium.com/efficient-way-to-read-large-postgresql-table-with-python-934d3edfdcc
i = 0
data_file = open('corpus.txt', 'w', encoding='UTF-8')
with connection.cursor(name='SO_posts_cursor') as cursor:
    cursor.itersize = 3000  # chunk size
    query = 'SELECT processedtitle, processedbody from postsmin;'
    cursor.execute(query)

    for row in cursor:
        title, body = row[0], row[1]
        if title:
            line = f"{title}\n"
            data_file.write(line)
        if body:
            line = f"{body}\n"
            data_file.write(line)

        i += 1
        if i % 25000 == 0:
            print(f"{str(i)} rows processed")

data_file.close()
end = datetime.now()
print("-" * 10)
print("elapsed time %s" % (end - start))
