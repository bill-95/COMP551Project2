import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
import string
import nltk
from nltk.stem.porter import PorterStemmer

train_data = []
target_data = []
test_data = []

def tokenize(text):
	#print("Original")
	#print(text)
	text = ''.join([ch for ch in text if ch not in string.punctuation])
	text = ''.join(i for i in text if not i.isdigit())
	text = ' '.join(word for word in text.split() if len(word)>4)
	#print("Processed")
	#print(text)
	#print()
	tokens = nltk.word_tokenize(text)
	stems = []
	for item in tokens:
		stems.append(PorterStemmer().stem(item))
	return stems


def preprocess(text):
	return result

def average(list):
	sum = 0
	for i in list:
		sum += i
	return str(sum/len(list))

# with open('../train_in.csv', 'rt') as csvfile:
# 	reader = csv.reader(csvfile, delimiter=',')
# 	next(reader)
# 	for row in reader:
# 		train_data.append(row[1])

# with open('../train_out.csv', 'rt') as csvfile:
# 	reader = csv.reader(csvfile, delimiter=',')
# 	next(reader)
# 	for row in reader:
# 		target_data.append(row[1])
with open('../train2.txt', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		train_data.append(row[1])

with open('../target.txt', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter='\n')
	next(reader)
	for row in reader:
		target_data.append(row[0])

with open('../test_in.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		test_data.append(row[1])


tf = TfidfVectorizer(tokenizer = tokenize, analyzer='word', ngram_range=(1,1), min_df = 1, stop_words = 'english',max_features= 60000, lowercase = True)
#tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 60000, lowercase = False)
X_train_counts = tf.fit_transform(train_data)

clf = MultinomialNB(alpha=0.1).fit(X_train_counts, target_data)

X_test_counts = tf.transform(test_data)
predicted = clf.predict(X_test_counts)
scores = cross_val_score(clf, X_train_counts, target_data, cv=7)

# print(tf.get_feature_names())
#print(tf.get_params())
#print(tf.get_stop_words())
# print(len(tf.get_feature_names()))
# print(len(tf.get_params()))

print(scores)
print(average(scores))
# print(clf.score(X_train_counts, target_data))

# TypeError: only integer arrays with one element can be converted to an index
f = open('moreData.csv','w')
f.write("id,category\n")
for i, row in enumerate(predicted):
    f.write(str(i)+","+predicted[i]+"\n")