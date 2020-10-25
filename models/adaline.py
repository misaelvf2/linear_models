import numpy as np


class Adaline:
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

    def train(self):
        # Initialize training variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.weights = np.array([0 for _ in range(num_features)]).reshape(num_features, 1)
        self.threshold = 0

        # Main loop
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            # Compute weighted sum
            weighted_sum = np.dot(self.weights.T, self.data) + self.threshold

            # Compute weight changes
            weight_changes = (1 / num_examples) * np.dot(self.data, (self.labels - weighted_sum).T)
            threshold_change = (1 / num_examples) * np.sum(self.data - weighted_sum)

            # Update weights
            self.weights = self.weights + self.learning_rate * weight_changes
            self.threshold = self.threshold + self.learning_rate * threshold_change

            # Pass through activation function
            classifier = np.vectorize(self.signum)
            output = classifier(weighted_sum)

            # Compute training error
            training_error = self.compute_training_error(output)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            print("Classification error: ", training_error)

    def test(self, data, labels):
        # Compute weighted sum
        weighted_sum = np.dot(self.weights.T, data) + self.threshold

        # Pass through activation function
        classifier = np.vectorize(self.signum)
        output = classifier(weighted_sum)

        # Compute testing error
        print("Classification error: ", self.compute_testing_error(output, labels))

    def signum(self, x):
        return 1 if x >= 0 else -1

    def multi_train(self):
        # Initialize multiclass labels
        self.initialize_multiclass_labels()

        # Initialize variables
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        cls_weights = {cls: None for cls in self.classes}
        cls_weighted_sum = {cls: None for cls in self.classes}
        cls_output = {cls: None for cls in self.classes}

        # Initialize weights
        for cls in self.classes:
            weights = np.array([0 for _ in range(num_features)]).reshape(num_features, 1)
            threshold = 0
            cls_weights[cls] = (weights, threshold)

        # Main loop over all classes
        last_error = 1.0
        same_error = 0
        while same_error < self.stopping_condition:
            for cls in self.classes:
                # Initialize labels
                labels = self.multicls_labels[cls]

                # Compute weighted sum
                weighted_sum = np.dot(cls_weights[cls][0].T, self.data) + cls_weights[cls][1]
                cls_weighted_sum[cls] = weighted_sum

                # Compute weight changes
                weight_changes = (1 / num_examples) * np.dot(self.data, (labels - weighted_sum).T)
                threshold_change = (1 / num_examples) * np.sum(labels - weighted_sum)

                # Update weights
                weights = cls_weights[cls][0] + self.learning_rate * weight_changes
                threshold = cls_weights[cls][1] + self.learning_rate * threshold_change
                cls_weights[cls] = (weights, threshold)

                # Pass through activation function
                activation = np.vectorize(self.signum)
                output = activation(weighted_sum)
                cls_output[cls] = output

            # Classify
            classifications = self.multi_classify(cls_output)

            # Compute training error
            training_error = self.compute_training_error(classifications)
            if training_error == last_error:
                same_error += 1
            else:
                same_error = 0
            last_error = training_error
            print("Classification error: ", training_error)

    def multi_classify(self, output):
        classifications = [(None, -np.inf) for _ in range(self.data.shape[1])]
        for cls, arr in output.items():
            for i, value in enumerate(arr[0]):
                if value > classifications[i][1]:
                    classifications[i] = (cls, value)
        classifications = np.array([cls for cls, _ in classifications]).reshape(1, -1)
        return classifications

    def initialize_multiclass_labels(self):
        for cls in self.classes:
            self.multicls_labels[cls] = np.where(self.raw_data['class'] == cls, 1, -1)

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

    def report_training_stats(self):
        print(self.training_stats)

    def report_testing_stats(self):
        print(self.testing_stats)

    def get_training_error(self):
        return self.training_stats['error']

    def get_testing_error(self):
        return self.testing_stats['error']
