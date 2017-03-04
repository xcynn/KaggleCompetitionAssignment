import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn import svm
from collections import Counter


train = pd.read_csv("train.csv")
# print train.head()
# print train.describe()

# Cross Validation
alg = svm.SVC(gamma=0.01, C=50.)
if True:
    train_samples = train[:]
    kf = KFold(train_samples.shape[0], n_folds=3, random_state=1)
    predictions = []
    for train_index, test_index in kf:
        print train_index, test_index
        # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
        train_set = (train_samples.iloc[train_index, 4:])
        # print train_set
        # exit(0)

        # The target we're using to train the algorithm.
        train_target = train_samples['Prediction'].iloc[train_index]
        alg.fit(train_set, train_target)

        # We can now make predictions on the test fold
        test_predictions = alg.predict(train_samples.iloc[test_index, 4:])
        predictions.append(test_predictions)

    predictions = np.concatenate(predictions, axis=0)
    # print predictions
    accuracy = float(len(predictions[predictions == train_samples['Prediction']]))/len(predictions)
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
