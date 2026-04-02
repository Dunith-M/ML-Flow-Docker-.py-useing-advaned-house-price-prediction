import pandas as pd
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        df = pd.read_csv(self.config.raw_data_path)
        return df

    def separate_features_target(self, df: pd.DataFrame):
        target_column = self.config.target_column

        # Split X and y
        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    def identify_feature_types(self, X: pd.DataFrame):
        # Numerical features
        num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Categorical features
        cat_features = X.select_dtypes(include=["object"]).columns.tolist()

        return num_features, cat_features

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        return X_train, X_test, y_train, y_test

    def initiate_data_transformation(self):
        # Step 1: Load data
        df = self.load_data()

        # Step 2: Separate features & target
        X, y = self.separate_features_target(df)

        # Step 3: Identify feature types
        num_features, cat_features = self.identify_feature_types(X)

        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "num_features": num_features,
            "cat_features": cat_features,
        }