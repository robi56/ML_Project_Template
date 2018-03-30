import json


class MLResult(object):
    def __init__(self,
                  data_size= None,
	              hyper_parameters=None,
                 missing_value_algorithm=None,
                 feature_list=None,
                 labels=None,
                 precision=None,
                 recall=None,
                 accuracy=None,
                 regression_coefficient=None,
                 confusion_matrix=None,
                 mean_squared_error=None,
                 variance_error=None,
                 comment=None):
        if data_size:
            self.data_size = data_size

        if hyper_parameters:
            self.hyper_parameters = hyper_parameters

        if missing_value_algorithm:
            self.missing_value_algorithm = missing_value_algorithm

        if feature_list:
            self.feature_list = feature_list

        if labels:
            self.labels = labels

        if precision:
            self.precision = precision

        if recall:
            self.recall = recall

        if accuracy:
            self.accuracy = accuracy

        if confusion_matrix is not None:
            self.confusion_matrix = confusion_matrix

        if regression_coefficient is not None:
            self.regression_coefficient=regression_coefficient

        if mean_squared_error:
            self.mean_squared_error= mean_squared_error

        if variance_error is not None:
            self.variance_error = variance_error
        if comment:
            self.comment = comment

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def toJSONInFile(self, filepath=None):
        if filepath:
            # Writing JSON data
            with open(filepath, 'w') as f:
                json.dump(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, fp=f)
