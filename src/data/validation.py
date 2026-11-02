import argparse
import sys

import great_expectations as gx
import pandas as pd


def build_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="credit_data_suite")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="default", value_set=[0, 1])
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="LIMIT_BAL"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", min_value=0, max_value=1_000_000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=[1, 2, 3, 4])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=[1, 2, 3])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2])
    )
    return suite


def validate_data(csv_path: str) -> bool:
    context = gx.get_context(mode="file")
    suite = build_suite()
    suite = context.suites.add_or_update(suite)
    print(f"[validation] Suite '{suite.name}' сохранён")
    data = pd.read_csv(csv_path)
    data_source = context.data_sources.add_or_update_pandas(name="credit_source")
    data_asset = data_source.add_dataframe_asset(name="credit_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(
            name="credit_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="credit_data_checkpoint",
            validation_definitions=[validation_definition],
        )
    )
    result = checkpoint.run(batch_parameters={"dataframe": data})

    success = result.success
    print(f"[validation] Результат: {'Success' if success else 'Failure'}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()

    if not validate_data(args.csv_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
