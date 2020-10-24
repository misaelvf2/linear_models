import numpy as np


class LogisticRegression:
    def __init__(self, data, labels, step_size):
        self.data = data
        self.labels = labels
        self.step_size = step_size
        self.weights = None
        self.threshold = None
        self.training_predictions = None
        self.training_results = None
        self.testing_predictions = None
        self.testing_results = None

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
