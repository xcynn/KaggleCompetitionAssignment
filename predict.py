import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn import svm
from collections import Counter


train = pd.read_csv("train.csv")
# print train.head()
# print train.describe()

# Customized validation
alg = svm.SVC(gamma=0.01, C=50.)
if True:
    train_samples = train[:]

    # manually form 3 set for cross validation
    train_index_set = []
    test_index_set = []
    word_count = 0
    for index, row in train_samples.iterrows():
        if row['Position'] == 1:
            # word is complete
            word_count += 1

        if (word_count % 3) == 0:
            test_index_set.append(index)
        else:
            train_index_set.append(index)

    print 'len(train_index_set)=%s, len(test_index_set)=%s' % (len(train_index_set), len(test_index_set))
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_set = (train_samples.iloc[train_index_set, 4:])
    # print train_set
    # exit(0)

    # The target we're using to train the algorithm.
    train_target = train_samples['Prediction'].iloc[train_index_set]
    alg.fit(train_set, train_target)

    # We can now make predictions on the test fold
    test_predictions = alg.predict(train_samples.iloc[test_index_set, 4:])
    # print list(test_predictions)
    # print list(train_samples.iloc[test_index_set]['Prediction'])

    # print test_predictions == train_samples.iloc[test_index_set]['Prediction']
    # print train_samples.iloc[test_index_set][test_predictions == train_samples.iloc[test_index_set]['Prediction']]
    accuracy = float(len(train_samples.iloc[test_index_set][test_predictions == train_samples.iloc[test_index_set]['Prediction']]))/len(test_index_set)
    print accuracy
    exit(0)

# Load test
test = pd.read_csv("test.csv")

# Train for test prediction
alg.fit(train.ix[:, 4:], train['Prediction'])
predictions = alg.predict(test.ix[:, 4:])
test_predictions = pd.DataFrame({
        "Id": test["Id"],
        "Prediction": predictions
    })
# Combine with train set for submission
train_predictions = train[['Id', 'Prediction']]
all_predictions = train_predictions.append(test_predictions)
# all_predictions = all_predictions.sort_values(by='Id')  # No need to sort
all_predictions.to_csv("submission_all.csv", index=False)
