import json
from pathlib import Path
import pandas as pd

from ..config.configuration import ConfigurationManager
from ..entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = config.schema

    # ----------------------------
    # 1. Load data
    # ----------------------------
    def load_data(self) -> pd.DataFrame:
        path = Path(self.config.raw_data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        return pd.read_csv(path)

    # ----------------------------
    # 2. Schema validation
    # ----------------------------
    def validate_columns(self, df: pd.DataFrame) -> dict:
        required_cols = set(self.schema["required_columns"])
        actual_cols = set(df.columns)

        missing_cols = list(required_cols - actual_cols)

        return {
            "missing_columns": missing_cols,
            "status": len(missing_cols) == 0
        }

    # ----------------------------
    # 3. Data type validation
    # ----------------------------
    def validate_dtypes(self, df: pd.DataFrame) -> dict:
        dtype_issues = {}

        for col in self.schema["columns"]:
            col_name = col["name"]
            expected_dtype = col["dtype"]

            if col_name not in df.columns:
                continue

            actual_dtype = str(df[col_name].dtype)

            if expected_dtype == "category":
                # Accept object/category
                if actual_dtype not in ["object", "category"]:
                    dtype_issues[col_name] = actual_dtype
            elif expected_dtype == "float":
                # Accept both integer and float numeric columns
                if actual_dtype not in ["float64", "float32", "int64", "int32"]:
                    dtype_issues[col_name] = actual_dtype
            elif expected_dtype == "int":
                if actual_dtype not in ["int64", "int32"]:
                    dtype_issues[col_name] = actual_dtype
            else:
                if expected_dtype not in actual_dtype:
                    dtype_issues[col_name] = actual_dtype

        return {
            "dtype_issues": dtype_issues,
            "status": len(dtype_issues) == 0
        }

    # ----------------------------
    # 4. Missing value validation
    # ----------------------------
    def validate_missing(self, df: pd.DataFrame) -> dict:
        issues = {}

        for col, threshold in self.schema["null_thresholds"].items():
            if col not in df.columns:
                continue

            null_ratio = df[col].isnull().mean()

            if null_ratio > threshold:
                issues[col] = float(null_ratio)

        return {
            "null_issues": issues,
            "status": len(issues) == 0
        }

    # ----------------------------
    # 5. Duplicate handling
    # ----------------------------
    def handle_duplicates(self, df: pd.DataFrame) -> tuple:
        keys = self.schema["primary_key"]

        before = len(df)
        df_cleaned = df.drop_duplicates(subset=keys)
        after = len(df_cleaned)

        return df_cleaned, {
            "duplicates_removed": before - after,
            "status": True
        }

    # ----------------------------
    # 6. Constraint validation
    # ----------------------------
    def validate_constraints(self, df: pd.DataFrame) -> dict:
        issues = {}

        constraints = self.schema.get("constraints", {})

        for col, rules in constraints.items():
            if col not in df.columns:
                continue

            min_val = rules.get("min", None)
            max_val = rules.get("max", None)

            invalid_count = 0

            if min_val is not None:
                invalid_count += (df[col] < min_val).sum()

            if max_val is not None:
                invalid_count += (df[col] > max_val).sum()

            if invalid_count > 0:
                issues[col] = int(invalid_count)

        return {
            "constraint_issues": issues,
            "status": len(issues) == 0
        }

    # ----------------------------
    # 7. Save validated data
    # ----------------------------
    def save_data(self, df: pd.DataFrame):
        path = Path(self.config.validated_data_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=False)

    # ----------------------------
    # 8. Save validation report
    # ----------------------------
    def save_report(self, report: dict):
        path = Path(self.config.report_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report, f, indent=4)

    # ----------------------------
    # 9. Run full validation pipeline
    # ----------------------------
    def run(self) -> dict:
        df = self.load_data()

        col_check = self.validate_columns(df)
        dtype_check = self.validate_dtypes(df)
        null_check = self.validate_missing(df)

        df, duplicate_info = self.handle_duplicates(df)

        constraint_check = self.validate_constraints(df)

        # Final status
        overall_status = all([
            col_check["status"],
            dtype_check["status"],
            null_check["status"],
            constraint_check["status"],
        ])

        report = {
            "columns": col_check,
            "dtypes": dtype_check,
            "missing_values": null_check,
            "duplicates": duplicate_info,
            "constraints": constraint_check,
            "overall_status": overall_status
        }

        # Save outputs
        self.save_data(df)
        self.save_report(report)

        if not overall_status:
            raise ValueError("Data validation failed. Check validation_report.json")

        return report


def main() -> None:
    config = ConfigurationManager()
    validation_config = config.get_data_validation_config()
    validation = DataValidation(validation_config)
    report = validation.run()
    print(f"Data validation completed. Overall status: {report['overall_status']}")
    print(f"Validated data saved to: {validation_config.validated_data_path}")
    print(f"Validation report saved to: {validation_config.report_file_path}")


if __name__ == "__main__":
    main()
