# -*- coding: utf-8 -*-

import numpy
import pandas
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


# Environment variables definition
Data_location="../Data/"
FileName = "fake.csv"
Language = "english"
InputFile = Data_location + FileName

# Internal variables
LoadedFile = pandas.read_csv(InputFile)
Stemmer = SnowballStemmer(Language)
Stopwords = set(nltk.corpus.stopwords.words(Language))
tokenizer = RegexpTokenizer(r'[a-z]+\w*')
TextData = []

#print("Importing the data...")
#Extract the text from the loaded CSV File
counter = 0
LoadedFile['text'] = LoadedFile['text'].str.lower()
LoadedFile['text'] = LoadedFile['text'].apply(nltk.word_tokenize)
print(LoadedFile['text'])
