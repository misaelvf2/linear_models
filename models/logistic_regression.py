import random
import math


class LogisticRegression:
    """
    Class implementing logistic regression for classification problems
    """
    def __init__(self, data, features, step_size):
        """
        Initializes parameters
        :param data: DataFrame
        :param features: List
        :param step_size: Float
        """
        self.data = data
        self.features = features
        self.weights = {}
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
        while same_error_count != 20:
            # Initialize weight changes to 0
            weight_changes = {k: 0 for k, _ in zip(self.features, range(len(self.features)))}
            weight_changes['threshold'] = 0
            # Compute outputs
            self.data['weighted_sum_output'] = \
                self.data.loc[:, 'clump_thickness':'mitoses'].apply(self.weighted_sum, axis=1)
            self.data['sigmoid_output'] = self.data['weighted_sum_output'].apply(self.sigmoid)
            # Compute weight changes
            self.data.loc[:, 'clump_thickness':'sigmoid_output'].\
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
        return 1 / (1 + math.exp(-x))

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
