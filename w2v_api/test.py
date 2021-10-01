import requests
import json


def main():
    r = requests.get("http://localhost:5001/")

    print('Checking that the server is running')
    if r.content != b'Welcome to our Machine Learning REST API!':
        raise Exception("Perhaps the server is not running?")

    print('\n\nSimilarity between two sentences in Di Sorbo example')
    payload = {
        'b': "the leaky bucket algorith fails in limiting the band-width",
        'a': 'We could use a leaky bucket algorithm to limit the band-width'
    }
    r = requests.get("http://localhost:5001/w2v/similarity", params=payload)
    print(r.content)


    print('\n\nEmbedding vector for a simple sentence')
    payload = {
        'txt': "The application needs to obtain the Bugzilla's custom fields and values such that it can use them to query and aggregate bugs based on the custom fields and their values."
    }
    r = requests.get("http://localhost:5001/w2v/embeddings", params=payload)
    print(r.content)

    print('\n\nSimilarity between two sentences - 1st example')
    payload = {
        'b': "The application needs to obtain the Bugzilla's custom fields and values such that it can use them to query and aggregate bugs based on the custom fields and their values.",
        'a': 'Some methods do not require you to log in.'
    }
    r = requests.get("http://localhost:5001/w2v/similarity", params=payload)
    print(r.content)

    print('\n\nSimilarity between two sentences - 2nd example')
    payload = {
        'b': "Left to right swap Increase the month",
        'a': "How to increase swap space? - Ask Ubuntu"
    }
    r = requests.get("http://localhost:5001/w2v/similarity", params=payload)
    print(r.content)

    print('\n\nSimilarity between two sentences - 3rd example')
    payload = {
        'b': "How does ArrayAdapter getView() method works?",
        'a': "Explanation of the getView() method of an ArrayAdapter"
    }
    r = requests.get("http://localhost:5001/w2v/similarity", params=payload)
    print(r.content)

if __name__ == '__main__':
    main()
