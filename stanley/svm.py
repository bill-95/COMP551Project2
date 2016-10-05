import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import string
train_data = []
target_data = []
test_data = []

#exclude = set(string.punctuation)

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

# parameters = {'clf__alpha': (1e-2, 1e-3, 1e-3),'clf__n_iter': (2, 4, 5,6,8)}

#text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),  Wasn't effective
# text_clf = Pipeline([('vect', CountVectorizer()),
#                       ('tfidf', TfidfTransformer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 40000)),
#                       ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, n_iter=2, random_state=42)),
# ])

text_clf = Pipeline([('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', max_features= 40000, lowercase = False)),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=4, random_state=42)),
])

_ = text_clf.fit(train_data, target_data)

# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(train_data, target_data)
# for param_name in sorted(parameters.keys()):
# 	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

predicted = text_clf.predict(test_data)

scores = cross_val_score(text_clf, train_data, target_data, cv=5)
print(scores)

# f = open('predictionsSvm.csv','w')
# f.write("id,category\n")
# for i, row in enumerate(predicted):
#     f.write(str(i)+","+predicted[i]+"\n")
# f.close()