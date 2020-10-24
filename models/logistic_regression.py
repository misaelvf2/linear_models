import random
import math
import numpy as np


class LogisticRegression:
    """
    Class implementing logistic regression for classification problems
    """
    def __init__(self, data, features, classes, step_size):
        """
        Initializes parameters
        :param data: DataFrame
        :param features: List
        :param classes: List
        :param step_size: Float
        """
        self.data = data
        self.features = features
        self.classes = classes
        self.weights = {}
        self.weights_multiclass = {cls: {} for cls in classes}
        self.step_size = step_size
        self.training_stats = {
            'Correct': 0,
            'Incorrect': 0,
            'Total': 0,
            'Success Rate': 0.0,
        }

    def train(self):
        """
        Trains model on data
        :return: None
        """
        # Initialize weights to random values
        random.seed(5)
        for feature in self.features:
            self.weights[feature] = random.uniform(-0.01, 0.01)
        self.weights['threshold'] = random.uniform(-0.01, 0.01)

        # Optimize
        previous_error = math.inf
        same_error_count = 0
        while same_error_count != 10:
            # Initialize weight changes to 0
            weight_changes = {k: 0 for k, _ in zip(self.features, range(len(self.features)))}
            weight_changes['threshold'] = 0
            # Compute outputs
            self.data['weighted_sum_output'] = \
                self.data.loc[:, self.features[0]:self.features[-1]].apply(self.weighted_sum, axis=1)
            self.data['sigmoid_output'] = self.data['weighted_sum_output'].apply(self.sigmoid)
            # Compute weight changes
            self.data.loc[:, self.features[0]:'sigmoid_output'].\
                apply(self.compute_weight_changes, weight_changes=weight_changes, axis=1)
            # Apply weight changes
            for weight in self.weights.keys():
                self.weights[weight] += self.step_size * weight_changes[weight]
            misclassification_error = self.compute_misclassification_error()
            if misclassification_error == previous_error:
                same_error_count += 1
            else:
                same_error_count = 0
            previous_error = misclassification_error

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function
        :param x: Float
        :return: Float
        """
        return 1 / (1 + np.exp(-x))

    def weighted_sum(self, x):
        """
        Performs weighted sum of features
        :param x: Series
        :return: Float
        """
        result = 0
        x = x.to_dict()
        for feature, value in x.items():
            result += value * self.weights[feature]
        result += self.weights['threshold']
        return result

    @staticmethod
    def compute_weight_changes(x, weight_changes):
        """
        Computes changes to feature weights
        :param x: Series
        :param weight_changes: Dict
        :return: None
        """
        x = x.to_dict()
        rt = x['class']
        y = x['sigmoid_output']
        del x['class']
        del x['weighted_sum_output']
        del x['sigmoid_output']
        for feature, value in x.items():
            weight_changes[feature] += (rt - y) * value
        weight_changes['threshold'] += rt - y

    def compute_misclassification_error(self):
        """
        Computes misclassification error
        :return: Int
        """
        false_positives = self.data[(self.data['sigmoid_output'] >= 0.5) & (self.data['class'] == 0)]
        false_negatives = self.data[(self.data['sigmoid_output'] < 0.5) & (self.data['class'] == 1)]
        return len(false_positives) + len(false_negatives)

    def report_training_stats(self):
        """
        Reports training statistics
        :return: None
        """
        true_positives = self.data[(self.data['sigmoid_output'] >= 0.5) & (self.data['class'] == 1)]
        true_negatives = self.data[(self.data['sigmoid_output'] < 0.5) & (self.data['class'] == 0)]
        self.training_stats['correct'] = len(true_positives) + len(true_negatives)
        self.training_stats['incorrect'] = len(self.data) - self.training_stats['correct']
        self.training_stats['total'] = len(self.data)
        self.training_stats['success_rate'] = self.training_stats['correct'] / self.training_stats['total']
        print(f"Correct: {self.training_stats['correct']}\n"
              f"Incorrect: {self.training_stats['incorrect']}\n"
              f"Total: {self.training_stats['total']}\n"
              f"Success Rate: {self.training_stats['success_rate']}\n")

    def return_incorrect_examples(self):
        print(f"False positives: {len(self.data[(self.data['sigmoid_output'] >= 0.5) & (self.data['class'] == 0)])}")
        print(f"False negatives: {len(self.data[(self.data['sigmoid_output'] < 0.5) & (self.data['class'] == 1)])}")
        return self.data[((self.data['sigmoid_output'] >= 0.5) & (self.data['class'] == 0)) |
                         ((self.data['sigmoid_output'] < 0.5) & (self.data['class'] == 1))]

    def __str__(self):
        return f"Data: {self.data}\nFeatures: {self.features}\nStep Size: {self.step_size}"

    def train_multiclass(self):
        """
        Trains model on data for multiclass datasets
        :return: None
        """
        # Initialize weights to random values
        random.seed(5)
        for cls in self.classes:
            for feature in self.features:
                self.weights_multiclass[cls][feature] = random.uniform(-0.01, 0.01)
            self.weights_multiclass[cls]['threshold'] = random.uniform(-0.01, 0.01)

        # Optimize
        previous_error = math.inf
        same_error_count = 0
        iterations = 50
        i = 0
        while i < iterations:
            # Initialize weight changes to 0
            weight_changes = {cls: {} for cls in self.classes}
            for cls in self.classes:
                weight_changes[cls] = {k: 0 for k, _ in zip(self.features, range(len(self.features)))}
                weight_changes[cls]['threshold'] = 0

            # Debugging
            # print(self.weights_multiclass[1]['ri'])

            # Compute outputs
            for cls in self.classes:
                self.data[f'{cls}_weighted_sum_output'] = \
                    self.data.loc[:, self.features[0]:self.features[-1]].apply(self.weighted_sum_multiclass, cls=cls, axis=1)
            for cls in self.classes:
                self.data[str(cls)] = self.data.apply(self.softmax, chosen_cls=cls, axis=1)

            # Compute weight changes
            for cls in self.classes:
                self.data.loc[:, self.features[0]:str(cls)].\
                    apply(self.compute_weight_changes_multiclass, cls=cls, weight_changes=weight_changes, axis=1)

            # Apply weight changes
            for cls in self.classes:
                weights = self.weights_multiclass[cls]
                for weight in weights.keys():
                    self.weights_multiclass[cls][weight] += self.step_size * weight_changes[cls][weight]

            # Compute misclassification error
            misclassification_error = self.compute_misclassification_error_multiclass()
            print(f"Correct: {misclassification_error}")
            if misclassification_error == previous_error:
                same_error_count += 1
            else:
                same_error_count = 0
            previous_error = misclassification_error
            i += 1

    def compute_weight_changes_multiclass(self, x, cls, weight_changes):
        """
        Computes changes to feature weights
        :param x: Series
        :param weight_changes: Dict
        :return: None
        """
        x = x.to_dict()
        rt = x['class']
        y = x[str(cls)]
        del x['class']
        del x[f'{cls}_weighted_sum_output']
        del x[str(cls)]
        for feature, value in x.items():
            if feature in self.features:
                weight_changes[cls][feature] += (rt - y) * value
        weight_changes[cls]['threshold'] += rt - y
        # print(weight_changes[1]['ri'])

    def weighted_sum_multiclass(self, x, cls):
        """
        Performs weighted sum of features
        :param x: Series
        :return: Float
        """
        result = 0
        x = x.to_dict()
        for feature, value in x.items():
            result += value * self.weights_multiclass[cls][feature]
        result += self.weights_multiclass[cls]['threshold']
        return result

    def softmax(self, x, chosen_cls):
        """
        Sigmoid function
        :param x: Float
        :param chosen_cls: Str
        :return: Float
        """
        denominator = 0
        for cls in self.classes:
            if cls != chosen_cls:
                denominator += np.exp(x[f'{cls}_weighted_sum_output'])
        # if chosen_cls == 1:
        #     print(chosen_cls, math.exp(x[f'{chosen_cls}_weighted_sum_output']) / denominator)
        return np.exp(x[f'{chosen_cls}_weighted_sum_output']) / denominator

    def compute_misclassification_error_multiclass(self):
        """
        Computes misclassification error
        :return: Int
        """
        output_columns = [str(cls) for cls in self.classes]
        self.data['prediction'] = self.data[output_columns].idxmax(axis=1)
        self.data['prediction'] = self.data['prediction'].astype('int32')
        correct_predictions = self.data[self.data['prediction'] == self.data['class']]
        # print(self.data[['class', '1', '2', '3', 'prediction']])
        # print(correct_predictions)
        return len(correct_predictions)