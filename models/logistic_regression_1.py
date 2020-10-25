import numpy as np


class LogisticRegression:
    def __init__(self, data, labels, classes, step_size, stopping_condition, raw_data=None):
        self.data = data
        self.labels = labels
        self.classes = classes
        self.step_size = step_size
        self.stopping_condition = stopping_condition
        self.weights = None
        self.threshold = None
        self.training_predictions = None
        self.training_results = None
        self.testing_predictions = None
        self.testing_results = None

        # Multiclass
        self.raw_data = raw_data
        self.discriminants = None

    def train(self):
        # Initialize weights
        np.random.seed(5)
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
        self.threshold = np.random.uniform(-0.01, 0.01)

        # Main loop; vectorized over all examples
        max_iter = 20
        for i in range(max_iter):
            # Compute weighted sums
            weighted_sums = np.dot(self.weights.T, self.data) + self.threshold

            # Compute sigmoids
            sigmoids = 1 / (1 + np.exp(-weighted_sums))

            # Compute weighted sum derivatives
            weighted_sum_derivatives = sigmoids - self.labels

            # Compute weight changes
            weight_changes = (1 / num_examples) * np.dot(self.data, weighted_sum_derivatives.T)
            threshold_change = (1 / num_examples) * np.sum(weighted_sum_derivatives)

            # Update weights
            self.weights -= self.step_size * weight_changes
            self.threshold -= self.step_size * threshold_change

            # Determine classification error
            print(self.misclassification_error(sigmoids))

    def test(self, data, labels):
        # Compute weighted sums
        weighted_sums = np.dot(self.weights.T, data) + self.threshold

        # Compute sigmoids
        sigmoids = 1 / (1 + np.exp(-weighted_sums))

        # Determine classification error
        print(self.testing_error(sigmoids, labels))

    def misclassification_error(self, sigmoids):
        classifier = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.training_predictions = classifier(sigmoids)
        self.training_results = self.training_predictions == self.labels
        correct = np.count_nonzero(self.training_results == True)
        incorrect = np.count_nonzero(self.training_results == False)
        success_rate = correct / (correct + incorrect)
        return success_rate

    def testing_error(self, sigmoids, labels):
        classifier = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.testing_predictions = classifier(sigmoids)
        self.testing_results = self.testing_predictions == labels
        correct = np.count_nonzero(self.testing_predictions == True)
        incorrect = np.count_nonzero(self.testing_results == False)
        success_rate = correct / (correct + incorrect)
        return success_rate

    def multi_train(self):
        # Initialize weights
        np.random.seed(5)
        num_features = self.data.shape[0]
        num_examples = self.data.shape[1]
        self.discriminants = {k: None for k in self.classes}
        for cls in self.discriminants.keys():
            weights = np.random.uniform(-0.01, 0.01, num_features).reshape(num_features, 1)
            threshold = np.random.uniform(-0.01, 0.01)
            self.discriminants[cls] = (weights, threshold)

        # Main loop; vectorized over all examples
        max_iter = 250
        for i in range(max_iter):
            # Initialize variables
            cls_weighted_sums = {k: None for k in self.classes}
            cls_softmax = {k: None for k in self.classes}
            cls_weighted_sum_derivatives = {k: None for k in self.classes}
            cls_weight_changes = {k: None for k in self.classes}
            cls_threshold_change = {k: None for k in self.classes}

            for cls in self.classes:
                # Compute weighted sums
                weighted_sums = np.dot(self.discriminants[cls][0].T, self.data) + self.discriminants[cls][1]
                cls_weighted_sums[cls] = weighted_sums

            for curr_cls in self.classes:
                # # Compute softmax
                denominator = 0
                for cls in self.classes:
                    if cls != curr_cls:
                        # cls_weighted_sums[cls] = normalize(cls_weighted_sums[cls], axis=1, norm='l1')
                        denominator += np.exp(cls_weighted_sums[cls])
                # cls_weighted_sums[curr_cls] = normalize(cls_weighted_sums[curr_cls], axis=1, norm='l1')
                cls_softmax[curr_cls] = np.divide(np.exp(cls_weighted_sums[curr_cls]), denominator)

                # Compute weighted sum derivatives
                cls_weighted_sum_derivatives[curr_cls] = cls_softmax[curr_cls] - self.labels

                # Compute weight changes
                cls_weight_changes[curr_cls] = (1 / num_examples) * np.dot(self.data, cls_weighted_sum_derivatives[curr_cls].T)
                cls_threshold_change[curr_cls] = (1 / num_examples) * np.sum(cls_weighted_sum_derivatives[curr_cls])

                # Update weights
                self.discriminants[curr_cls] = (self.discriminants[curr_cls][0] - self.step_size * cls_weight_changes[curr_cls],
                                           self.discriminants[curr_cls][1] - self.step_size * cls_threshold_change[curr_cls])

            # Determine classification error
            print(self.multi_training_error(cls_softmax))

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
