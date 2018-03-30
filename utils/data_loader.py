import codecs
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from sklearn import preprocessing
from tensorflow.contrib.learn.python.learn.datasets import base



class DataSet(object):

    def __init__(self,
                 patient_profile,
                 labels,
                 one_hot=False):

        if one_hot==True:
            num_labels = len(np.unique(labels))
            labels = np.eye(num_labels)[labels]

        self._num_examples = len(patient_profile)
        self._patent_profiles = patient_profile
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._all_x = None
        self._all_y = None

    @property
    def patient_profiles(self):
        return self._patent_profiles

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._patent_profiles = self._patent_profiles[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._patent_profiles[start:end], self._labels[start:end]





def get_cancer_data(file_url,output_column_index,test_size=0.30, random_state=42):

    if file_url is not None:
        dataframe=pd.read_csv(file_url)


        data = dataframe.iloc[:, 2:]
        labels = dataframe.iloc[:, 1]


        #normalization
        #data = (data - dataframe.mean()) / (data.max() - data.min())


        data = data.as_matrix(columns=None)
        labels = np.squeeze(np.asarray(labels))
        # categorical to numerical
        le = preprocessing.LabelEncoder()
        labels =le.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.40, random_state=42)
        print(np.shape(X_train))
        return [X_train, X_test, y_train, y_test]




def get_cancer_data_as_object(file_url,output_column_index,test_size=0.30, random_state=42):
    train_data, test_data, train_labels, test_labels = get_cancer_data(file_url,output_column_index,test_size, random_state)

    train = DataSet(train_data, train_labels, one_hot=True)
    test = DataSet(test_data, test_labels, one_hot=True)
    return base.Datasets(train=train, validation=None, test=test)
