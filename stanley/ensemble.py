import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
import string
import nltk
from nltk.stem.porter import PorterStemmer

train_data = []
target_data = []
test_data = []

def tokenize(text):
    text = ''.join(i for i in text if not i.isdigit())
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def average(list):
	sum = 0
	for i in list:
		sum += i
	return str(sum/len(list))

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


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 60000, lowercase = False)
X_train_counts = tf.fit_transform(train_data)
# fsel = ExtraTreesClassifier()
# fsel = fsel.fit(X_train_counts, target_data)
# model = SelectFromModel(fsel, prefit=True)
# X_train_counts = model.transform(X_train_counts)
# clf = MultinomialNB(alpha=0.1).fit(X_train_counts, target_data)

clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty='l2', loss='squared_hinge', dual=False,
                       tol=1e-3))),
  ('classification', RandomForestClassifier())
])
clf.fit(X_train_counts, target_data)

# X_test_counts = tf.transform(test_data)
# predicted = clf.predict(X_test_counts)
scores = cross_val_score(clf, X_train_counts, target_data, cv=7)
print(scores)
print(average(scores))


# print(clf.score(X_train_counts, target_data))



# f = open('predictionsMaxFeatures.csv','w')
# f.write("id,category\n")
# for i, row in enumerate(predicted):
#     f.write(str(i)+","+predicted[i]+"\n")
# f.close()