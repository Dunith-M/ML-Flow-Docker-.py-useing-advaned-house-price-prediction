import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        df = pd.read_csv(self.config.raw_data_path)
        return df

    def separate_features_target(self, df: pd.DataFrame):
        target_column = self.config.target_column

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    def identify_feature_types(self, X: pd.DataFrame):
        num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
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

    # Numerical Pipeline
    def get_numeric_pipeline(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    # Categorical Pipeline
    def get_categorical_pipeline(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                ),
            ]
        )

    # NEW: Combine pipelines
    def get_preprocessor(self, num_features, cat_features):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.get_numeric_pipeline(), num_features),
                ("cat", self.get_categorical_pipeline(), cat_features),
            ]
        )
        return preprocessor

    def initiate_data_transformation(self):
        # 1. Load data
        df = self.load_data()

        # 2. Separate features and target
        X, y = self.separate_features_target(df)

        # 3. Identify feature types
        num_features, cat_features = self.identify_feature_types(X)

        # 4. Train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 5. Build preprocessor
        preprocessor = self.get_preprocessor(num_features, cat_features)

        # 6. Fit ONLY on training data
        X_train_transformed = preprocessor.fit_transform(X_train)

        # 7. Transform test data
        X_test_transformed = preprocessor.transform(X_test)

        # 8. Save preprocessor
        os.makedirs(self.config.preprocessor_path, exist_ok=True)

        preprocessor_file_path = os.path.join(
            self.config.preprocessor_path,
            "preprocessor.pkl"
        )

        joblib.dump(preprocessor, preprocessor_file_path)

        return {
            "X_train": X_train_transformed,
            "X_test": X_test_transformed,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor_path": preprocessor_file_path,
        }