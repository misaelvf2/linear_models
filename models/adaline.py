import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    """
    Implementation of the Adaline algorithm
    """
    def __init__(self, data, labels, classes, learning_rate, threshold, stopping_condition, raw_data=None):
        """
        Initializes class
        :param data: numpy.ndarray
        :param labels: numpy.ndarray
        :param classes: List
        :param learning_rate: Float
        :param threshold: Float
        :param stopping_condition: Float
        :param raw_data: DataFrame
        """
        # Initialization variables
        self.data = data
        self.labels = labels
        self.classes = classes
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.stopping_condition = stopping_condition
        self.raw_data = raw_data

        # Training variables
        self.weights = None
        self.bias = None
        self.classifications = None

        # Training statistics
        self.training_stats = dict(correct=0, incorrect=0, classified=0, false_positives=0, false_negatives=0,
                                   true_positives=0, true_negatives=0, error=0.0, accuracy=0.0)
        self.errors = []

        # Testing statistics
        self.testing_stats = dict(correct=0, incorrect=0, classified=0, false_positives=0, false_negatives=0,
                                   true_positives=0, true_negatives=0, error=0.0, accuracy=0.0)

        # Multiclass variables
        self.multicls_labels = dict()
        self.cls_weights = dict()
        self.cls_bias = dict()
        self.multi_classifications = None

    def train(self):
        """
        Main training loop
        :return: None
        """
        # Initialize training variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.weights = np.array([0 for _ in range(num_features)]).reshape(num_features, 1)
        self.bias = 0

        # Main loop
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Compute weighted sum
            weighted_sum = np.dot(self.weights.T, self.data) + self.bias

            # Compute weight changes
            weight_changes = (1 / num_examples) * np.dot(self.data, (self.labels - weighted_sum).T)
            bias_change = (1 / num_examples) * np.sum(self.data - weighted_sum)
            # print(f"Gradient calculation: Avg({1 / num_examples}) * Dot({np.dot(self.data, (self.labels - weighted_sum).T)}) = Result({weight_changes})")

            # Update weights
            # print("Weights before: ", self.weights)
            self.weights = self.weights + self.learning_rate * weight_changes
            self.bias = self.bias + self.learning_rate * bias_change
            # print("Weight changes: ", weight_changes)
            # print("Weights after: ", self.weights)

            # Pass through activation function
            classifier = np.vectorize(self.signum)
            output = classifier(weighted_sum)
            self.classifications = output

            # Compute training error
            training_error = self.compute_training_error(output)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            self.errors.append(training_error)
            print(training_error)

    def test(self, data, labels):
        """
        Tests given data
        :param data: numpy.ndarray
        :param labels: numpy.ndarray
        :return: None
        """
        # Compute weighted sum
        weighted_sum = np.dot(self.weights.T, data) + self.bias

        # Pass through activation function
        classifier = np.vectorize(self.signum)
        output = classifier(weighted_sum)
        self.classifications = output

        # Compute testing error
        testing_error = self.compute_testing_error(output, labels)
        print(testing_error)

    def signum(self, x):
        """
        Implements signum activation function
        :param x: Float
        :return: Int
        """
        return 1 if x >= self.threshold else -1

    def multi_train(self):
        """
        Main training loop for multiclass problems
        :return: None
        """
        # Initialize multiclass labels
        self.initialize_multiclass_labels()

        # Initialize variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        cls_weighted_sum = {cls: None for cls in self.classes}
        cls_output = {cls: None for cls in self.classes}

        # Initialize weights
        for cls in self.classes:
            weights = np.array([0 for _ in range(num_features)]).reshape(num_features, 1)
            bias = 0
            self.cls_weights[cls] = (weights, bias)

        # Main loop over all classes
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            for cls in self.classes:
                # Initialize labels
                labels = self.multicls_labels[cls]

                # Compute weighted sum
                weighted_sum = np.dot(self.cls_weights[cls][0].T, self.data) + self.cls_weights[cls][1]
                cls_weighted_sum[cls] = weighted_sum

                # Compute weight changes
                weight_changes = (1 / num_examples) * np.dot(self.data, (labels - weighted_sum).T)
                bias_change = (1 / num_examples) * np.sum(labels - weighted_sum)

                # Update weights
                weights = self.cls_weights[cls][0] + self.learning_rate * weight_changes
                bias = self.cls_weights[cls][1] + self.learning_rate * bias_change
                self.cls_weights[cls] = (weights, bias)

                # Pass through activation function
                activation = np.vectorize(self.signum)
                output = activation(weighted_sum)
                cls_output[cls] = output

            # Classify
            classifications = self.multi_classify(self.data, cls_output)
            self.classifications = classifications

            # Compute training error
            training_error = self.compute_training_error(classifications)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            self.errors.append(training_error)
            print(training_error)

    def multi_test(self, data, labels):
        """
        Classifies given data in multiclass case
        :param data: numpy.ndarray
        :param labels: numpy.ndarray
        :return: None
        """
        # Initialize variables
        cls_weighted_sum = {cls: None for cls in self.classes}
        cls_output = {cls: None for cls in self.classes}

        for cls in self.classes:
            # Compute weighted sum
            weighted_sum = np.dot(self.cls_weights[cls][0].T, data) + self.cls_weights[cls][1]
            cls_weighted_sum[cls] = weighted_sum

            # Pass through activation function
            activation = np.vectorize(self.signum)
            output = activation(weighted_sum)
            cls_output[cls] = output

        # Classify
        classifications = self.multi_classify(data, cls_output)
        self.classifications = classifications

        # Compute testing error
        testing_error = self.compute_testing_error(classifications, labels)
        print(testing_error)

    def multi_classify(self, data, output):
        """
        Classifies weighted sums in multiclass case
        :param data: numpy.ndarray
        :param output: numpy.ndarray
        :return: List
        """
        classifications = [(None, -np.inf) for _ in range(data.shape[1])]
        for cls, arr in output.items():
            for i, value in enumerate(arr[0]):
                if value > classifications[i][1]:
                    classifications[i] = (cls, value)
        classifications = np.array([cls for cls, _ in classifications]).reshape(1, -1)
        return classifications

    def initialize_multiclass_labels(self):
        """
        Separates out class labels in multiclass case
        :return: None
        """
        for cls in self.classes:
            self.multicls_labels[cls] = np.where(self.raw_data['class'] == cls, 1, -1)

    def compute_training_error(self, output):
        """
        Computes training error
        :param output: numpy.ndarray
        :return: Float
        """
        results = output == self.labels  # Maybe make this an instance variable
        self.training_stats['correct'] = np.count_nonzero(results == True)
        self.training_stats['incorrect'] = np.count_nonzero(results == False)
        self.training_stats['classified'] = self.training_stats['correct'] + self.training_stats['incorrect']
        self.training_stats['accuracy'] = self.training_stats['correct'] / self.training_stats['classified']
        self.training_stats['error'] = 1 - self.training_stats['accuracy']
        return self.training_stats['error']

    def compute_testing_error(self, output, labels):
        """
        Computes testing error
        :param output: numpy.ndarray
        :param labels: numpy.ndarray
        :return: Float
        """
        results = output == labels
        self.testing_stats['correct'] = np.count_nonzero(results == True)
        self.testing_stats['incorrect'] = np.count_nonzero(results == False)
        self.testing_stats['classified'] = self.testing_stats['correct'] + self.testing_stats['incorrect']
        self.testing_stats['accuracy'] = self.testing_stats['correct'] / self.testing_stats['classified']
        self.testing_stats['error'] = 1 - self.testing_stats['accuracy']
        return self.testing_stats['error']

    def report_training_stats(self):
        """
        Prints training stats
        :return: None
        """
        print(self.training_stats)

    def report_testing_stats(self):
        """
        Prints testing stats
        :return: None
        """
        print(self.testing_stats)

    def get_training_error(self):
        """
        Returns training error
        :return: Float
        """
        return self.training_stats['error']

    def get_testing_error(self):
        """
        Returns testing error
        :return: Float
        """
        return self.testing_stats['error']

    def plot_error(self):
        """
        Plots testing error with respect to number of iterations
        :return: None
        """
        plt.plot(self.errors)
        plt.ylabel('Error')
        plt.savefig("error.png")
        # plt.show()

    def report_classifications(self):
        """
        Prints model classifications
        :return:
        """
        print("Classifications: ", self.classifications)
