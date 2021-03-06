import numpy as np
import preprocessing
from models.logistic_regression import LogisticRegression
from models.adaline import Adaline
from sklearn.model_selection import StratifiedKFold

def main(model, dataset, learning_rate, stopping_condition, threshold=None):
    """
    Main program
    :param model: Str
    :param dataset: Str
    :param learning_rate: Float
    :param stopping_condition: Float
    :param threshold: Float
    :return: None
    """
    # Import, process, and normalize data
    if dataset == 'breast':
        df = preprocessing.process_breast_cancer_data()
    elif dataset == 'glass':
        df = preprocessing.process_glass_data()
    elif dataset == 'iris':
        df = preprocessing.process_iris_data()
    elif dataset == 'soybean':
        df = preprocessing.process_soybean_data()
    elif dataset == 'voter':
        df = preprocessing.process_voter_data()

    # Set up stratified 5-fold cross-validation; only necessary for classificaton
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    training_sets, test_sets = [], []
    for fold, (train, test) in enumerate(skf.split(X=np.zeros(len(df)), y=df.iloc[:, -1:])):
        training_sets.append(df.iloc[train])
        test_sets.append(df.iloc[test])

    # Train; run 5 experiments in total
    training_errors, trained_models = [], []
    for training_set in training_sets:
        print("\nTraining:")
        training_data = training_set.iloc[:, 1:-1].to_numpy().T
        training_labels = training_set.iloc[:, -1:].to_numpy().T
        classes = df['class'].unique()
        if model == 'adaline':
            my_model = \
                Adaline(training_data, training_labels, classes, learning_rate, threshold, stopping_condition, raw_data=training_set)
        elif model == 'logistic_regression':
            my_model = \
                LogisticRegression(training_data, training_labels, classes, learning_rate, threshold, stopping_condition, raw_data=training_set)
        if dataset == 'breast' or dataset == 'voter':
            my_model.train()
        elif dataset == 'glass' or dataset == 'iris' or dataset == 'soybean':
            my_model.multi_train()
        trained_models.append(my_model)
        training_errors.append(my_model.get_training_error())
        my_model.plot_error()
        my_model.report_classifications()

    # Test; run 5 experiments in total
    testing_errors = []
    for model, test_set in zip(trained_models, test_sets):
        print("\nTesting: ")
        testing_data = test_set.iloc[:, 1:-1].to_numpy().T
        testing_labels = test_set.iloc[:, -1:].to_numpy().T
        if dataset == 'breast' or dataset == 'voter':
            model.test(testing_data, testing_labels)
        elif dataset == 'glass' or dataset == 'iris' or dataset == 'soybean':
            model.multi_test(testing_data, testing_labels)
        testing_errors.append(model.get_testing_error())
        model.report_classifications()

    # Report average results
    average_training_error = sum(training_errors) / len(training_errors)
    average_testing_error = sum(testing_errors) / len(testing_errors)
    print("\nSummary:")
    print(f"Average training error: {average_training_error}")
    print(f"Average testing error: {average_testing_error}")

# main('adaline', 'breast', 0.01, 10, threshold=0)
main('logistic_regression', 'iris', 0.01, 10, threshold=0.5)
