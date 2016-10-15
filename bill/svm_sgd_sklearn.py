#! python3
# coding: utf-8


import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


target_names = []
target = []
data = []
target_dict = {'math': 0, 'cs': 1, 'stat': 2, 'physics': 3}
testData = []
#load training data
with open('train_in_modified.csv') as infile:
    for i, row in enumerate(csv.reader(infile)):
        if i == 0:
            continue
        if row[1] == "abstract":
            continue
            
        data.append(row[1])
        target.append(target_dict[row[2]])
        target_names.append(row[2])
        
target_names = set(target_names)
#load test data
with open('test_in.csv') as infile:
    for i, row in enumerate(csv.reader(infile)):
        if i == 0:
            continue
        testData.append(row[1])


pipeline = Pipeline([
    ('vect',  CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

outfile = open('predictions_svm.csv', 'w')
writer = csv.writer(outfile)
pred_map = {0: 'math', 1: 'cs', 2: 'stat', 3: 'physics'}
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (1.0,),
    #'vect__max_features': (None, 5000, 10000, 30000),
    'vect__ngram_range': ((1, 2),),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l2',),
    'clf__alpha': (0.00001,),
    'clf__penalty': ('l2',),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])

    grid_search.fit(data, target)

    predictions = grid_search.predict(testData)
    for y in predictions:
        writer.writerow([pred_map[y]])
    
    print("Best score: %0.3f" % grid_search.best_score_)

outfile.close()


