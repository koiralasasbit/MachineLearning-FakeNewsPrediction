# This is a sample Python script.

# importing the required dependencies

import numpy as np
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import contractions
import nltk

# disabling SSL check!
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')
nltk.download('wordnet')
# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('train.csv')

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
print(news_dataset['content'])

print(contractions.fix(news_dataset['content'][0]))
print(news_dataset['content'][0])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def expandContraction(content):
    """Expand contractions in the content"""
    return contractions.fix(content)


def to_lower(content):
    """Convert every characters in the content to lower case"""
    return content.lower()


def remove_punctuation(content):
    """Remove punctuations from the content"""
    return re.sub('[%s]' % re.escape(string.punctuation), '', content)


def remove_digit(content):
    """Remove words and digits containing digits from the content"""
    return re.sub('\w*\d\w*', '', content)


def remove_stopwords(content):
    """Remove stopwords from the content"""
    stop_words = stopwords.words('english')
    return " ".join([word for word in content.split() if word not in stop_words])


def remove_email(content):
    """Rephrase email addresses into 'emailadd' in the content"""
    return re.sub('\S+@\S+(?:\.\S+)+', 'emailadd', content)


def remove_url(content):
    """Rephrase urls into 'urladd' in the content"""
    return re.sub('https?://\S+|www\.\S+', 'urladd', content)


def stem_words(content):
    """Stem words in the content"""
    return " ".join([stemmer.stem(word) for word in content.split()])


def lemmatize_words(content):
    """lemmatize words in the content"""
    return " ".join([lemmatizer.lemmatize(word) for word in content.split()])


def preprocess(content):
    # Replace apostrophe with single quote
    content = content.replace("’", "'")
    # Expand Contractions
    content = expandContraction(content)
    # Lower Case
    content = to_lower(content)
    # Remove Punctuations
    content = remove_punctuation(content)
    # Remove words and digits containing digits
    content = remove_digit(content)
    # Remove Stopwords
    content = remove_stopwords(content)
    # Email and Url removal
    content = remove_email(content)
    content = remove_url(content)
    # Stemming
    content = stem_words(content)
    # Lemmatization
    content = lemmatize_words(content)

    return content


news_dataset['content'] = news_dataset['content'].apply(preprocess)

print(news_dataset['content'][0])

X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X.shape)
# Feature Extraction (TF-IDF Vectorization)
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X.shape)


