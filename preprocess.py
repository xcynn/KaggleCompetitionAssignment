import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn import svm
from collections import Counter

train = pd.read_csv("train.csv")


def count_words(index):
        dataset = train.iloc[index]
        words = []
        word = ''
        for index, row in dataset.iterrows():
            if row['Position'] == 1 and word:
                words.append(word)
                word = row['Prediction']
            else:
                word += row['Prediction']
        words.append(word)  # include the last word

        word_counter = Counter(words)
        print 'number of words:', len(word_counter.keys())
        print word_counter
        return word_counter

# Count the total train dataset in terms of words
# count_words(train.index)


# # Show that KFold is not suitable for cross validation in this case
# kf = KFold(train.shape[0], n_folds=8, random_state=42)
# for i, (train_index, test_index) in enumerate(kf):
#     print 'Cross validation No.%s' % i

#     train_word_set = set(count_words(train_index).keys())
#     test_word_set = set(count_words(test_index).keys())
#     print 'test_word_set - train_word_set =', test_word_set - train_word_set

# Split into my own train and validation set
train_index = []
test_index = []
word_count = 0
for index, row in train.iterrows():
    if row['Position'] == 1:
        # word is complete
        word_count += 1

    if (word_count % 3) == 0:
        test_index.append(index)
    else:
        train_index.append(index)

train_word_set = set(count_words(train_index).keys())
test_word_set = set(count_words(test_index).keys())
print 'test_word_set - train_word_set =', test_word_set - train_word_set

print 'done'
