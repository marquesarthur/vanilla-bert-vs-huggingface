

# Docker Word2Vec REST API

This is a docker image that takes two sentences and returns the cosine similarity between their vector representations.

The model uses the `word2vec-google-news-300` trained dataset. 

A sentence is converted into its vector representation following the algorithm (pesudo-code):

```python
def vector(sentence):
    model = gensim word2vec-google-news-300
    c = 0
    vector = numpy.zeros(model.size) 
    for word in sentence:
        if c in model.vocab:
            vector += model.word
            c += 1
    
    return numpy.divide(vector, c)
            

def similarity(a, b):
    a_vec = vector(a)
    b_vec = vector(b)
    return cosine_similarity(a_vec, b_vec)
```

First, we convert the sentences to their vector representations. Then, we compute the cosine similarity between them.
To convert a sentence, we check whether a word exists in the model vocabulary. If it does, we retrieve the vector representation of the word from the model.
The final vector is computing summing all vectors of the words that exist in the model and diving the number of words that exist in the model.
The division assists in accounting for lengthy sentences. 

The final implementation uses [Aerim Kim](https://www.linkedin.com/in/aerinykim/)'s [python code](https://bitbucket.org/yunazzang/aiwiththebest_byor/src/master/)

## Setup

### From docker hub

```shell script
docker pull msarthur/w2v-rest:latest
```

### From source

```shell script

git clone git@github.com:marquesarthur/w2v-rest-api.git
cd w2v-rest-api
wget -O SO_vectors_200.bin https://zenodo.org/record/1199620/files/SO_vectors_200.bin
docker build -t msarthur/w2v-rest .
```


## Publish image

```shell script
docker push msarthur/w2v-rest:latest
```

___

## RUN

Remember to remove old containers

```shell script
docker rm w2v-rest
docker run --name w2v-rest -p 5001:5001 msarthur/w2v-rest
```

## Testing
```shell script
curl -XGET 'http://localhost:5001/w2v/similarity?a=Have+you+ever+tried+to+used+the+method+string.split%28%22hello+world%22%29&b=try+using+the+string.split+method' | json_pp
curl -XGET 'http://localhost:5001/w2v/similarity?a=Obama%20speaks%20to%20the%20media%20in%20Illinois&b=The%20president%20greets%20the%20press%20in%20Chicago' | json_pp
curl -XGET 'http://localhost:5001/w2v/similarity?a=Obama%20speaks%20to%20the%20media%20in%20Illinois&b=Having%20a%20tough%20time%20finding%20an%20orange%20juice%20press%20machine%3F' | json_pp

curl -XGET 'http://localhost:5001/w2v/similarity?a=these%20are%20very%20similar%20phrases&b=phrases%20that%20are%20very%20similar' | json_pp
```

Response should look like:

```json
{
   "a" : "Have you ever tried to used the method string.split(\"hello world\")",
   "b" : "try using the string.split method",
   "err" : null,
   "sim" : "0.54513955"
}
``` 

You can also run the [test file](test.py) and test the API programmatically


# Docker FastText REST API

Command to setup a docker Postgre container found [here](https://sclorg.github.io/containers-doc/postgresql/9.4/).
A second example is available [here](https://www.optimadata.nl/blogs/1/n8dyr5-how-to-run-postgres-on-docker-part-1)

```shell scrit
docker pull postgres:9.4
```


```shell scrit
docker run -d -e POSTGRES_USER=w2v -e POSTGRES_PASSWORD=password123 --name w2v-postgres -p 5432:5432  postgres:9.4
```

FastText model build using [Crokage](https://github.com/muldon/CROKAGE-replication-package) [pre-processed data dump](http://lascam.facom.ufu.br/companion/crokage/dump2018crokagereplicationpackage.backup)








## Currently supported models

* SO_vectors_200.bin - see [Word Embeddings for the Software Engineering Domain](https://zenodo.org/record/1199620#.YFEFlv4TGhg) `default`
* word2vec-google-news-300 - requires tweaking `app.py`, look for a `FIXME` tag

*note 1* : Ideally, all models could be loaded in memory at different endpoints. Nonetheless, this requires a significant amount of computational resources. I don't have these, so I made the image only load the `default` model

*note 2* : More models and implementations in the near future !!!