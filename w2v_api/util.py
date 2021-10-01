import nltk

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
import string

from nltk.tokenize import word_tokenize


def clean_text(text):
    for c in string.punctuation:
        if c not in ["'", "′", "’"]:
            text = text.replace(c, "")
    text = text.lower().replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("'ll", " will")
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]  # remove non-alphabatical words
    stops = set(stopwords.words("english"))
    word_filter = [w for w in words if not w in stops and len(w) >= 3]
    text_return = " ".join(word_filter)
    for c in string.punctuation:
        text_return = text.replace(c, "")
    return text_return
