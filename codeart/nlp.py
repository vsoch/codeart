"""

Copyright (C) 2019-2020 Vanessa Sochat.

This Source Code Form is subject to the terms of the
Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

Modified from https://github.com/Visual-mov/Colorful-Julia (MIT License)

"""

from textblob import TextBlob, Word
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import *
import nltk
import nltk.data
import pandas
import gensim
import re


def remove_nonenglish_chars(text):
    return re.sub("[^a-zA-Z]", " ", text)


def text2sentences(text, remove_non_english_chars=True):
    try:
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    except:
        nltk.download("punkt")
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    if remove_non_english_chars:
        text = remove_nonenglish_chars(text)
    for s in tokenizer.tokenize(text):
        yield s


def processText(text):
    """combines text2sentences and sentence2words"""
    vector = []
    for line in text2sentences(text):
        words = sentence2words(line)
        vector = vector + words
    return vector


def sentence2words(sentence, remove_stop_words=False, lower=True):
    """This is intended to transform a line of "code" words into words,
       meaning that we keep any word that is all characters or underscore,
       OR all special characters
    """
    if isinstance(sentence, list):
        sentence = sentence[0]

    # Split based on white spaces or periods
    re_white_space = re.compile("(\s+|[.])")
    if lower:
        sentence = sentence.lower()

    # Split by white space or period
    words = re_white_space.split(sentence.strip())
    words = [w for w in words if w not in [".", " "] and w]

    if remove_stop_words:
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words and w]

    return words
