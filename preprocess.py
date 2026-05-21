import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)