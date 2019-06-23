# -*- coding: utf-8 -*-

import numpy
from tqdm import tqdm
import pandas
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm

#==============================================================================================================
#						Initialization 
#==============================================================================================================

# Environment variables definition
Data_location="../Data/challenge/"
Language = "english"

trainBodiesFile = Data_location + "train_bodies.csv"
trainHeadlinesFile = Data_location + "train_stances.csv"
testBodiesFile = Data_location + "competition_test_bodies.csv"
testHeadlinesFile = Data_location + "competition_test_stances.csv"

# Loading data 
trainBodies = pandas.read_csv(trainBodiesFile)
trainHeadlines = pandas.read_csv(trainHeadlinesFile)
testBodies = pandas.read_csv(testBodiesFile)
testHeadlines = pandas.read_csv(testHeadlinesFile)

# Internal variables
Stemmer = SnowballStemmer(Language)
wnl = WordNetLemmatizer()
Stopwords = set(nltk.corpus.stopwords.words(Language))
tokenizer = RegexpTokenizer(r'[a-z]+\w*')
TextData = []

def separation(text):
	Separator = "+"*50
	print("")
	print("")
	print(Separator+" "+text+" "+Separator)
	print("")
	print("")

#==============================================================================================================
#					Making characters lower case 
#==============================================================================================================

separation("Making characters lower case")
def lower(textList) :
	counter = 0
	newTextList = []
	for text in tqdm(textList):
		newTextList.append(text.lower())

	return newTextList

print("Bodies")
trainBodies['articleBody'] = lower(trainBodies['articleBody'])
testBodies['articleBody'] = lower(testBodies['articleBody'])

print("")

print("headlines")
trainHeadlines['Headline'] =  lower(trainHeadlines['Headline'])
testHeadlines['Headline'] =  lower(testHeadlines['Headline'])


#==============================================================================================================
#					Tokenizing the data	
#==============================================================================================================


separation("Tokenizing the data")

def tokenization(textList):
	counter = 0
	TokenizedData = list()
	for text in tqdm(textList):
		TokenizedData.append(tokenizer.tokenize(text))
		counter = counter + 1

	return TokenizedData

print("Bodies")
trainBodies['articleBody'] = tokenization(trainBodies['articleBody'])
testBodies['articleBody'] = tokenization(testBodies['articleBody'])

print("")

print("Headlines")
trainHeadlines['Headline'] =  tokenization(trainHeadlines['Headline'])
testHeadlines['Headline'] =  tokenization(testHeadlines['Headline'])



#==============================================================================================================
#					Removing stopwords		
#==============================================================================================================

separation("Removing stopwords")

def rem_stopwords(wordlistList) :
	newList = list()
	for wordlist in tqdm(wordlistList):
		tempList = list()
		for word in wordlist :
			if word not in Stopwords:
				tempList.append(word)

		newList.append(tempList)

	return newList

print("Bodies")
trainBodies['articleBody'] = rem_stopwords(trainBodies['articleBody'])
testBodies['articleBody'] = rem_stopwords(testBodies['articleBody'])

print("")

print("Headlines")
trainHeadlines['Headline'] =  rem_stopwords(trainHeadlines['Headline'])
testHeadlines['Headline'] =  rem_stopwords(testHeadlines['Headline'])





#==============================================================================================================
#						Lemmatize Data		
#==============================================================================================================

separation("Lemmatize the data")

def convert(tag):
	if tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return None

def lemmatization(wordlistList) :
	result = []
	LemmatizedData = []
	counter = 0
	for wordlist in tqdm(wordlistList):
		#if (SanitizedText is not None) and (len(SanitizedText) is not 0):
		LocalList = []
		for word, pos in pos_tag(wordlist):
			wordnetPos = convert(pos) or wordnet.NOUN
			LocalList.append(wnl.lemmatize(word, pos=wordnetPos).lower())
		LemmatizedData.append(LocalList)

	return LemmatizedData

print("Bodies")
trainBodies['articleBody'] = lemmatization(trainBodies['articleBody'])
testBodies['articleBody'] = lemmatization(testBodies['articleBody'])

print("")

print("Headlines")
trainHeadlines['Headline'] =  lemmatization(trainHeadlines['Headline'])
testHeadlines['Headline'] =  lemmatization(testHeadlines['Headline'])


#==============================================================================================================
#					Preparing for vectorization	
#==============================================================================================================

separation("Preparing for vectorization")

print("Merging the data...")
trainingData = pandas.merge(trainHeadlines,trainBodies,how='inner',on=['Body ID', 'Body ID'])
testingData = pandas.merge(testHeadlines,testBodies,how='inner',on=['Body ID', 'Body ID'])


print("building training vocabulary")
trainSet = []
counter = 0 
for tokenizedText in tqdm(trainingData['Headline']):
	passed = False
	for word in tokenizedText :
		if passed == False:
			trainSet.append(word)
			passed = True
		else :
			trainSet[counter] = trainSet[counter] + " " + word

	counter = counter + 1
			
counter = 0
for tokenizedText in tqdm(trainingData['articleBody']):
	for word in tokenizedText :
		trainSet[counter] = trainSet[counter] + " " + word
	counter = counter + 1


trainSetSeries = pandas.Series(trainSet)

trainStances = list()
for stance in trainingData['Stance'] :
	trainStances.append(stance)


transformedTrainStances = [None] * len(trainStances)
count = 0
while (count<len(trainStances)) :
	if trainStances[count] == 'disagree' :
		transformedTrainStances[count] = 0
	elif trainStances[count] == 'agree' :
		transformedTrainStances[count] = 1
	elif trainStances[count] == 'discuss' :
		transformedTrainStances[count] = 1
	else :
		transformedTrainStances[count] = 0
	count = count + 1


print("building test vocabulary")
testSet = []
counter = 0 
for tokenizedText in tqdm(testingData['Headline']):
	passed = False
	for word in tokenizedText :
		if passed == False:
			testSet.append(word)
			passed = True
		else :
			testSet[counter] = testSet[counter] + " " + word

	counter = counter + 1
			
counter = 0
for tokenizedText in tqdm(testingData['articleBody']):
	for word in tokenizedText :
		testSet[counter] = testSet[counter] + " " + word
	counter = counter + 1

testSetSeries = pandas.Series(testSet)

testStances = list()
for stance in testingData['Stance'] :
	testStances.append(stance)

transformedTestStances = [None] * len(testStances)
count = 0
while (count<len(testStances)) :
	if testStances[count] == 'disagree' :
		transformedTestStances[count] = 0
	elif testStances[count] == 'agree' :
		transformedTestStances[count] = 1
	elif testStances[count] == 'discuss' :
		transformedTestStances[count] = 1
	else :
		transformedTestStances[count] = 0
	count = count + 1

#==============================================================================================================
#						vectorization	
#==============================================================================================================

separation("Vectorization")

vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 0.2) #Read about max_df and min_df

vectorizedTrainArticles = vectorizer.fit_transform(trainSetSeries).toarray()
vectorizedTestArticles = vectorizer.transform(testSetSeries).toarray()

#transformedTrainStances = transformedTrainStances.astype('int')
#transformedTestStances = transformedTestStances.astype('int')

print("vectorized Train Articles (Headline+body)")
print(vectorizedTrainArticles)



#==============================================================================================================
#						Training	
#==============================================================================================================

separation("Training")

#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
#clf.predict(X[:2, :])
#clf.predict_proba(X[:2, :])
#clf.score(X, y)


#Train the SVM
clf = svm.SVC()
clf.fit(vectorizedTrainArticles, transformedTrainStances)
svm.SVC(kernel = 'linear', probability = True, random_state = 0)




#==============================================================================================================
#						Prediction	
#==============================================================================================================

separation("Prediction")
predictions = clf.predict(vectorizedTestArticles)
a=numpy.array(transformedTestStances)

count = 0
for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count = count + 1

print(count/len(predictions))
