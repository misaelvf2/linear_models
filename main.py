import preprocessing
from models.logistic_regression import LogisticRegression

def main():
    # Import, process, and normalize breast cancer data
    # df = preprocessing.process_breast_cancer_data()
    df = preprocessing.process_glass_data()

    # Create model
    # my_model = LogisticRegression(df, list(df.columns[1:-1]), [0, 1], 0.05)
    my_model = LogisticRegression(df, list(df.columns[1:-1]), [1, 2, 3, 5, 6, 7], 0.001)

    # Train model
    my_model.train_multiclass()

    # Print trained model
    # my_model.report_training_stats()

    # print(my_model.return_incorrect_examples())

main()
