import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Class implementing Logistic Regression algorithm
    """
    def __init__(self, data, labels, classes, learning_rate, threshold, stopping_condition, raw_data=None):
        """
        Initialization
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

    def train(self):
        """
        Main training loop
        :return: None
        """
        # Initialize weights
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
        self.bias = np.random.uniform(-0.01, 0.01)

        # Main loop; vectorized over all examples -- Change to stopping condition
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Compute weighted sums
            weighted_sum = np.dot(self.weights.T, self.data) + self.bias

            # Compute sigmoids
            sigmoid = (1 / (1 + np.exp(-weighted_sum))) - 0.5

            # Compute weighted sum derivatives
            weighted_sum_derivatives = sigmoid - self.labels

            # Compute weight changes
            weight_changes = (1 / num_examples) * np.dot(self.data, weighted_sum_derivatives.T)
            bias_change = (1 / num_examples) * np.sum(weighted_sum_derivatives)

            # Update weights
            self.weights -= self.learning_rate * weight_changes
            self.bias -= self.learning_rate * bias_change

            # Pass through activation function
            classifier = np.vectorize(lambda x: 1 if x >= self.threshold else -1)
            output = classifier(sigmoid)
            self.classifications = output

            # Determine classification error
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
        # Compute weighted sums
        weighted_sum = np.dot(self.weights.T, data) + self.bias

        # Compute sigmoids
        sigmoid = (1 / (1 + np.exp(-weighted_sum))) - 0.5

        # Pass through activation function
        classifier = np.vectorize(lambda x: 1 if x >= self.threshold else -1)
        output = classifier(sigmoid)
        self.classifications = output

        # Determine classification error
        testing_error = self.compute_testing_error(output, labels)
        print(testing_error)

    def compute_training_error(self, output):
        """
        Computes training error
        :param output: numpy.ndarray
        :return: Float
        """
        results = output == self.labels
        self.training_stats['correct'] = np.count_nonzero(results == True)
        self.training_stats['incorrect'] = np.count_nonzero(results == False)
        self.training_stats['classified'] = self.training_stats['correct'] + self.training_stats['incorrect']
        self.training_stats['accuracy'] = self.training_stats['correct'] / self.training_stats['classified']
        self.training_stats['error'] = 1 - self.training_stats['accuracy']
        return self.training_stats['error']

    def compute_testing_error(self, output, labels):
        results = output == labels
        self.testing_stats['correct'] = np.count_nonzero(results == True)
        self.testing_stats['incorrect'] = np.count_nonzero(results == False)
        self.testing_stats['classified'] = self.testing_stats['correct'] + self.testing_stats['incorrect']
        self.testing_stats['accuracy'] = self.testing_stats['correct'] / self.testing_stats['classified']
        self.testing_stats['error'] = 1 - self.testing_stats['accuracy']
        return self.testing_stats['error']

    def initialize_multiclass_labels(self):
        """
        Separates out class labels in case of multiclass problems
        :return: None
        """
        for cls in self.classes:
            self.multicls_labels[cls] = np.where(self.raw_data['class'] == cls, 1, -1)

    def multi_train(self):
        """
        Main training loop for multiclass problems
        :return:
        """
        # Initialize multiclass labels
        self.initialize_multiclass_labels()

        # Initialize variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]

        # Initialize weights
        for cls in self.classes:
            weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
            bias = np.random.uniform(-0.01, 0.01)
            self.cls_weights[cls] = (weights, bias)

        # Main loop; vectorized over all examples
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Initialize variables
            cls_weighted_sum = {k: None for k in self.classes}
            cls_softmax = {k: None for k in self.classes}
            cls_weighted_sum_derivatives = {k: None for k in self.classes}
            cls_weight_changes = {k: None for k in self.classes}
            cls_bias_change = {k: None for k in self.classes}

            for cls in self.classes:
                # Compute weighted sums
                weighted_sums = np.dot(self.cls_weights[cls][0].T, self.data) + self.cls_weights[cls][1]
                cls_weighted_sum[cls] = weighted_sums

            for curr_cls in self.classes:
                # Compute softmax
                denominator = 0
                for cls in self.classes:
                    if cls != curr_cls:
                        denominator += np.exp(cls_weighted_sum[cls])
                cls_softmax[curr_cls] = np.divide(np.exp(cls_weighted_sum[curr_cls]) - 0.5, denominator - 0.5)

                # Compute weighted sum derivatives
                cls_weighted_sum_derivatives[curr_cls] = cls_softmax[curr_cls] - self.multicls_labels[curr_cls]

                # Compute weight changes
                cls_weight_changes[curr_cls] = (1 / num_examples) * np.dot(self.data, cls_weighted_sum_derivatives[curr_cls].T)
                cls_bias_change[curr_cls] = (1 / num_examples) * np.sum(cls_weighted_sum_derivatives[curr_cls])

                # Update weights
                self.cls_weights[curr_cls] = (self.cls_weights[curr_cls][0] - self.learning_rate * cls_weight_changes[curr_cls],
                                           self.cls_weights[curr_cls][1] - self.learning_rate * cls_bias_change[curr_cls])

            # Classify
            classifications = self.multi_classify(self.data, cls_softmax)
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
        Tests given data in multiclass problems
        :param data: numpy.ndarray
        :param labels: numpy.ndarray
        :return: None
        """
        # Initialize variables
        cls_weighted_sums = {k: None for k in self.classes}
        cls_softmax = {k: None for k in self.classes}

        for cls in self.classes:
            # Compute weighted sums
            weighted_sums = np.dot(self.cls_weights[cls][0].T, data) + self.cls_weights[cls][1]
            cls_weighted_sums[cls] = weighted_sums

        for curr_cls in self.classes:
            # Compute softmax
            denominator = 0
            for cls in self.classes:
                if cls != curr_cls:
                    denominator += np.exp(cls_weighted_sums[cls])
            cls_softmax[curr_cls] = np.divide(np.exp(cls_weighted_sums[curr_cls]) - 0.5, denominator - 0.5)

        # Classify
        classifications = self.multi_classify(data, cls_softmax)
        self.classifications = classifications

        # Compute testing error
        testing_error = self.compute_testing_error(classifications, labels)
        print(testing_error)

    def multi_classify(self, data, output):
        """
        Classifies output in multiclass case
        :param data: numpy.ndarray
        :param output: numpy.ndarray
        :return: numpy.ndarray
        """
        classifications = [(None, -np.inf) for _ in range(data.shape[1])]
        for cls, arr in output.items():
            for i, value in enumerate(arr[0]):
                if value > classifications[i][1]:
                    classifications[i] = (cls, value)
        classifications = np.array([cls for cls, _ in classifications]).reshape(1, -1)
        return classifications

    def multi_training_error(self, softmax):
        classifications = [(None, 0) for _ in range(self.data.shape[1])]
        for cls, arr in softmax.items():
            for i, value in enumerate(arr[0]):
                if value > classifications[i][1]:
                    classifications[i] = (cls, value)
        classifications = np.array([cls for cls, _ in classifications]).reshape(1, -1)
        results = classifications == self.labels
        correct = np.count_nonzero(results == True)
        incorrect = np.count_nonzero(results == False)
        success_rate = correct / (correct + incorrect)
        return success_rate

    def report_training_stats(self):
        """
        Prints trainign stats
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
        Plots error with respect to number of iterations
        :return: None
        """
        plt.plot(self.errors)
        plt.ylabel('Error')
        plt.savefig("error.png")

    def report_classifications(self):
        """
        Prints model classifications
        :return: 
        """
        print("Classifications: ", self.classifications)
