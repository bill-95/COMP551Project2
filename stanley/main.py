import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


train_data = []
with open('../train_in.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		train_data.append(row[1])


vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train_data)


target_data = []
with open('../train_out.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		target_data.append(row[1])

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
 ])



# meanVal = np.mean(train_data == target_data)
# print(meanVal) 


test_data = []
with open('../test_in.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		test_data.append(row[1])

clf = MultinomialNB().fit(X_train_counts, target_data)



X_test_counts = vectorizer.transform(test_data)
predicted = clf.predict(X_test_counts)
print(X_test_counts.shape)
print(predicted.shape)

f = open('predictions.csv','w')
f.write("id,category\n")
for i, row in enumerate(predicted):
    f.write(str(i)+","+predicted[i]+"\n")
f.close()