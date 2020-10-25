import numpy as np
import preprocessing
from models.logistic_regression_1 import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def main():
    # Import, process, and normalize breast cancer data
    df = preprocessing.process_glass_data()

    # Set up stratified 5-fold cross-validation; only necessary for classificaton
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    training_sets, test_sets = [], []
    for fold, (train, test) in enumerate(skf.split(X=np.zeros(len(df)), y=df.iloc[:, -1:])):
        training_sets.append(df.iloc[train])
        test_sets.append(df.iloc[test])

    # Train; run 5 experiments in total
    trained_models = []
    for training_set in training_sets:
        print("\nTraining:")
        training_data = training_set.iloc[:, 1:-1].to_numpy().T
        training_labels = training_set.iloc[:, -1:].to_numpy().T
        step_size = 0.085
        my_model = LogisticRegression(training_data, training_labels, [1, 2, 3, 5, 6, 7], step_size)
        my_model.multi_train()
        trained_models.append(my_model)

    # Test; run 5 experiments in total
    for model, test_set in zip(trained_models, test_sets):
        print("\nTesting: ")
        testing_data = test_set.iloc[:, 1:-1].to_numpy().T
        testing_labels = test_set.iloc[:, -1:].to_numpy().T
        model.multi_test(testing_data, testing_labels)

main()
