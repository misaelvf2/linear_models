import numpy as np


class LogisticRegression:
    def __init__(self, data, labels, classes, learning_rate, stopping_condition, raw_data=None):
        # Initialization variables
        self.data = data
        self.labels = labels
        self.classes = classes
        self.learning_rate = learning_rate
        self.stopping_condition = stopping_condition
        self.raw_data = raw_data

        # Training variables
        self.weights = None
        self.threshold = None

        # Training statistics
        self.training_stats = dict(correct=0, incorrect=0, classified=0, false_positives=0, false_negatives=0,
                                   true_positives=0, true_negatives=0, error=0.0, accuracy=0.0)

        # Testing statistics
        self.testing_stats = dict(correct=0, incorrect=0, classified=0, false_positives=0, false_negatives=0,
                                   true_positives=0, true_negatives=0, error=0.0, accuracy=0.0)

        # Multiclass variables
        self.multicls_labels = dict()
        self.cls_weights = dict()
        self.cls_threshold = dict()

    def train(self):
        # Initialize weights
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
        self.threshold = np.random.uniform(-0.01, 0.01)

        # Main loop; vectorized over all examples -- Change to stopping condition
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Compute weighted sums
            weighted_sum = np.dot(self.weights.T, self.data) + self.threshold

            # Compute sigmoids
            sigmoid = (1 / (1 + np.exp(-weighted_sum))) - 0.5

            # Compute weighted sum derivatives
            weighted_sum_derivatives = sigmoid - self.labels

            # Compute weight changes
            weight_changes = (1 / num_examples) * np.dot(self.data, weighted_sum_derivatives.T)
            threshold_change = (1 / num_examples) * np.sum(weighted_sum_derivatives)

            # Update weights
            self.weights -= self.learning_rate * weight_changes
            self.threshold -= self.learning_rate * threshold_change

            # Pass through activation function
            classifier = np.vectorize(lambda x: 1 if x >= 0 else -1)
            output = classifier(sigmoid)

            # Determine classification error
            training_error = self.compute_training_error(output)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            print(training_error)

    def test(self, data, labels):
        # Compute weighted sums
        weighted_sum = np.dot(self.weights.T, data) + self.threshold

        # Compute sigmoids
        sigmoid = (1 / (1 + np.exp(-weighted_sum))) - 0.5

        # Pass through activation function
        classifier = np.vectorize(lambda x: 1 if x >= 0 else -1)
        output = classifier(sigmoid)  #  Maybe make this an instance variable

        # Determine classification error
        testing_error = self.compute_testing_error(output, labels)
        print(testing_error)

    def compute_training_error(self, output):
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
        for cls in self.classes:
            self.multicls_labels[cls] = np.where(self.raw_data['class'] == cls, 1, -1)

    def multi_train(self):
        # Initialize multiclass labels
        self.initialize_multiclass_labels()

        # Initialize variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]

        # Initialize weights
        for cls in self.classes:
            weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
            threshold = np.random.uniform(-0.01, 0.01)
            self.cls_weights[cls] = (weights, threshold)

        # Main loop; vectorized over all examples
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Initialize variables
            cls_weighted_sum = {k: None for k in self.classes}
            cls_softmax = {k: None for k in self.classes}
            cls_weighted_sum_derivatives = {k: None for k in self.classes}
            cls_weight_changes = {k: None for k in self.classes}
            cls_threshold_change = {k: None for k in self.classes}

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
                cls_threshold_change[curr_cls] = (1 / num_examples) * np.sum(cls_weighted_sum_derivatives[curr_cls])

                # Update weights
                self.cls_weights[curr_cls] = (self.cls_weights[curr_cls][0] - self.learning_rate * cls_weight_changes[curr_cls],
                                           self.cls_weights[curr_cls][1] - self.learning_rate * cls_threshold_change[curr_cls])

            # Classify
            classifications = self.multi_classify(self.data, cls_softmax)

            # Compute training error
            training_error = self.compute_training_error(classifications)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            print(training_error)

    def multi_test(self, data, labels):
        # Initialize variables
        cls_weighted_sums = {k: None for k in self.classes}
        cls_softmax = {k: None for k in self.classes}

        for cls in self.classes:
            # Compute weighted sums
            weighted_sums = np.dot(self.discriminants[cls][0].T, data) + self.discriminants[cls][1]
            cls_weighted_sums[cls] = weighted_sums

        for curr_cls in self.classes:
            # Compute softmax
            denominator = 0
            for cls in self.classes:
                if cls != curr_cls:
                    denominator += np.exp(cls_weighted_sums[cls])
            cls_softmax[curr_cls] = np.exp(cls_weighted_sums[curr_cls]) / denominator

        # Determine classification error
        print(self.multi_testing_error(cls_softmax, labels))

    def multi_classify(self, data, output):
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

    def multi_testing_error(self, softmax, labels):
        classifications = [(None, 0) for _ in range(len(labels[0]))]
        for cls, arr in softmax.items():
            for i, value in enumerate(arr[0]):
                if value > classifications[i][1]:
                    classifications[i] = (cls, value)
        classifications = np.array([cls for cls, _ in classifications]).reshape(1, -1)
        results = classifications == labels
        correct = np.count_nonzero(results == True)
        incorrect = np.count_nonzero(results == False)
        success_rate = correct / (correct + incorrect)
        return success_rate

    def report_training_stats(self):
        print(self.training_stats)

    def report_testing_stats(self):
        print(self.testing_stats)

    def get_training_error(self):
        return self.training_stats['error']

    def get_testing_error(self):
        return self.testing_stats['error']
