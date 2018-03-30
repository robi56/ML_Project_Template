
import config
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.metrics import accuracy_score

from utils.statistics import MLResult
from utils.file_name_generator import get_filename
from utils.data_loader import get_cancer_data


data_file_url = os.path.join(config.BASE_DIR+  '/data/breast_cancer.csv')
train_data, test_data, train_labels, test_labels=get_cancer_data(data_file_url, output_column_index=1)

#Model Selection
svm_object = svm.SVC()
#Training

model = svm_object.fit(train_data, train_labels)

#Test

predicated_labels = model.predict(test_data)


# evaluation
score = accuracy_score(test_labels, predicated_labels, normalize=True)
confusion_matrix = metrics.confusion_matrix(test_labels,predicated_labels)
precision = average_precision_score(test_labels,predicated_labels)
recall = recall_score(test_labels,predicated_labels)


# publishing result

labels = ['Malignant', 'benign']
hyper_parameters = {'C':1}


statistics = MLResult(
                          hyper_parameters=hyper_parameters,
                          missing_value_algorithm=None,
                          feature_list=None,
                          labels=labels,
                          precision=precision,
                          recall=recall,
                          accuracy=score,
                          confusion_matrix=confusion_matrix.tolist())


# timestamp_score

filename = 'results/'+get_filename(score)
statistics.toJSONInFile(filepath=filename)

