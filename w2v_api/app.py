from flask import Flask, request, jsonify

print("Loading stopwords")

import nltk
from w2v_api.util import clean_text

nltk.download('stopwords')
nltk.download('punkt')

from w2v_api.embedding import Word2Vec, SOWord2Vec

print("Loading StackOverflow word2vec")
w2v = SOWord2Vec()
w2v.load(file_name="SO_vectors_200.bin")

# FIXME: uncomment for standard word2vec-google-news-300
# print("Loading Google News word2vec")
# w2v = Word2Vec()
# w2v.load(gensim_pre_trained_model="word2vec-google-news-300")


print("Init flask")
app = Flask(__name__)

# https://stackoverflow.com/questions/14888799/disable-console-messages-in-flask-server
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route("/", methods=['GET'])
def welcome():
    return "Welcome to our Machine Learning REST API!"


@app.route("/w2v/similarity", methods=['GET'])
def similarity_route():
    response = {
        "a": None,
        "b": None,
        "sim": None,
        "err": None
    }
    try:
        a = request.args.get("a")
        b = request.args.get("b")
        # y = clean_text(a)
        # x = clean_text(b)
        sim = w2v.similarity(a, b)
        response['a'] = a
        response['b'] = b
        response['sim'] = str(sim)
        return jsonify(response)
    except Exception as err:
        response['err'] = str(err)
        return jsonify(response)


@app.route("/w2v/embeddings", methods=['GET'])
def embedding_route():
    response = {
        "text": None,
        "embeddings": None,
        "err": None
    }
    try:
        a = request.args.get("txt")
        x = clean_text(a)

        response['text'] = a
        response['embeddings'] = str(w2v.vector(x))

        return jsonify(response)
    except Exception as err:
        response['err'] = str(err)
        return jsonify(response)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
