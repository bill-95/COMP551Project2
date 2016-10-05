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
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

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

# Straight up naive bayes
# vectorizer = CountVectorizer()
# X_train_counts = vectorizer.fit_transform(train_data)
# tfidf_transformer = TfidfTransformer()
# X_train_counts = tfidf_transformer.fit_transform(X_train_counts)

# clf = MultinomialNB().fit(X_train_counts, target_data)

# X_test_counts = vectorizer.transform(test_data)
# X_test_counts = tfidf_transformer.transform(X_test_counts)
# predicted = clf.predict(X_test_counts)

#added tfidf
# vectorizer = CountVectorizer()
# X_train_counts = vectorizer.fit_transform(train_data)
# tfidf_transformer = TfidfTransformer()
#X_train_counts = tfidf_transformer.fit_transform(X_train_counts)

#tf = TfidfVectorizer(tokenizer=tokenize, analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 40000, lowercase = False)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 40000, lowercase = False)
X_train_counts = tf.fit_transform(train_data)

#print(tf.get_feature_names())
#print(tf.get_params())
#print(tf.get_stop_words())

# print(len(tf.get_feature_names()))
# print(len(tf.get_params()))
clf = MultinomialNB(alpha=0.1).fit(X_train_counts, target_data)

#X_test_counts = vectorizer.transform(test_data)
#X_test_counts = tfidf_transformer.transform(X_test_counts)


X_test_counts = tf.transform(test_data)
predicted = clf.predict(X_test_counts)

scores = cross_val_score(clf, X_train_counts, target_data, cv=7)
print(scores)

# f = open('predictionsMaxFeatures.csv','w')
# f.write("id,category\n")
# for i, row in enumerate(predicted):
#     f.write(str(i)+","+predicted[i]+"\n")
# f.close()