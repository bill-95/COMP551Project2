import csv
import numpy as np
import string
from nltk.corpus import stopwords
import pickle

def createDictionary(list):
	print("Creating complete word frequency dictionary")
	print("May take a while (an hour)")
	exclude = set(string.punctuation)
	encodeTrainList = []
	words = {}
	for abstract in list:
		wordlist = abstract.split()
		correctedwordlist = []
		for word in wordlist:
			corrw = ''.join(ch for ch in word if ch not in exclude).lower()
			if(len(corrw) >3 & len(corrw) < 20):
				correctedwordlist.append(corrw)
		filtered_words = [word for word in correctedwordlist if word not in stopwords.words('english')]

		wordfreq = [(p, filtered_words.count(p)) for p in filtered_words]
		encodeTrainList.append(wordfreq)
		for k,v in wordfreq:
			if k in words:
				words[k]+=v
			else:
				words[k]=v
	pickle.dump(words, open( "freqDict.p", "wb" ) )
	pickle.dump(encodeTrainList, open( "encodeTrainList.p", "wb" ))
	return words, encodeTrainList

def seperateByCategory(train, target):
	classes = {}
	classes['math'] = {}
	classes['stat'] = {}
	classes['physics'] = {}
	classes['cs'] = {}
	for i, val in enumerate(train):
		for k,v in val:
			if k in classes[target[i]]:
				classes[target[i]][k] += v
			else:
				classes[target[i]][k] = v
	pickle.dump(classes, open( "separated.p", "wb" ))

	return classes

def prob(x, y, seperated, wordDict, wordDictSize):
	wordDictSize = len(wordDict)
	if x in seperated[y]:
		count = seperated[y][x]+1.0
	else:
		count = 1.0
	if x in wordDict:
		totalCount = wordDict[x] + wordDictSize
	else:
		totalCount = wordDictSize
	
	ret = (count/totalCount)
	if(ret == 0.0):
		print("incorrect "+str(count)+" "+str(totalCount))
	return ret

def test(text_x, classes, test_y):
	correct = 0
	for i, abstractPair in enumerate(test_x):
		result = []
		maxClass = "math"
		maxProb = 0.0
		for classif in classes: 
			totalProb = 1.0
			for k,v in abstractPair:
				probi = prob(k, classif, separated, wordDict, wordDictSize)
				totalProb *= probi
			result.append(totalProb)
			if(totalProb > maxProb):
	 			maxProb = totalProb
	 			maxClass = classif
		if(maxClass == test_y[i]):
			correct += 1
	print("correct: ")
	print(correct)
	print("total: ")
	print(len(test_y))

def forward(text_x, classes):
	output = []
	for i, abstractPair in enumerate(test_x):
		result = []
		maxClass = "math"
		maxProb = 0.0
		for classif in classes: 
			totalProb = 1.0
			for k,v in abstractPair:
				probi = prob(k, classif, separated, wordDict, wordDictSize)
				totalProb *= probi
			result.append(totalProb)
			if(totalProb > maxProb):
	 			maxProb = totalProb
	 			maxClass = classif
		output.append(maxClass)
	f = open('nbpredictions.csv','w')
	f.write("id,category\n")
	for i, row in enumerate(output):
	    f.write(str(i)+","+output[i]+"\n")


train_data = []
target_data = []
test_data = []



with open('../train_in.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		train_data.append(row[1])

with open('../train_out.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		target_data.append(row[1])


with open('../test_in.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		test_data.append(row[1])

wordDict, encodedTrainData = createDictionary(train_data)
#encodedTrainData = pickle.load( open( "encodeTrainList.p", "rb"))
#wordDict = pickle.load( open( "freqDict.p", "rb" ))

wordDictSize = 0
for k in wordDict:
	wordDictSize += wordDict[k]  

separated = seperateByCategory(encodedTrainData, target_data)
#separated = pickle.load( open( "separated.p", "rb"))

test_x = encodedTrainData[70000:]
test_y = target_data[70000:]

classes = ['math','stat','physics','cs']
test(test_x, classes, test_y)
forward(test_x, classes)