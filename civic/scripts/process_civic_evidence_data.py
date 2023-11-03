import sys

sys.path.append("../../../civic")
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from civic.utils.filesystem_utils import create_folder_if_not_exists
from config import PROJECT_ROOT

THE_CLASS_DISTRIBUTION_IS = "The class distribution is"


def remove_duplicates(data, column_combination):
    duplicated = data.duplicated(
        subset=column_combination,
        keep=False,
    )
    data = data.loc[~duplicated]
    return data


def do_data_cleaning(data):
    data = data.dropna(subset=["sourceAbstract", "evidenceLevel"], how="any")
    print(
        f"Dropping all records without abstract or evidenceLevel. {data.shape[0]} records remaining"
    )
    data = remove_duplicates(
        data,
        [
            "diseaseId",
            "significance",
            "molecularProfileId",
            "therapyIds",
            "sourceId",
        ],
    )
    print(
        f"Dropping non unique combinations of disease, significance, MP, therapy and abstract."
        f"{data.shape[0]} records remaining"
    )

    print("Extracting therapy combinations")
    data["therapyNames"] = data.therapyNames.map(
        lambda x: x.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(" ", "")
        .replace(",", "-")
    )
    data["therapyNames"] = data.therapyNames.map(lambda x: np.nan if x == "" else x)
    data = data.dropna(
        subset=["diseaseName", "significance", "molecularProfileName", "therapyNames"],
        how="all",
    )
    print(
        f"Dropping records that have none of diseaseName, significance, MP or therapy {data.shape[0]} records remaining"
    )
    print("Fill NaN with Unknown")
    data = data.fillna("Unknown")
    assert data.isna().sum().sum() == 0
    print("No NaN values left")
    data["prependString"] = (
        "DiseaseName: "
        + data.diseaseName
        + "\n"
        + "Molecular Profile Name: "
        + data.molecularProfileName
        + "\n"
        + "Therapies: "
        + data.therapyNames
        + "\n"
        + "Significance: "
        + data.significance
    )
    print("Created string to be prepended to abstract for uniqueness of input")
    assert data["prependString"].isna().sum() == 0
    n_duplicates = data.duplicated(
        subset=["prependString", "sourceId"], keep=False
    ).sum()
    assert n_duplicates == 0
    print("Every combination of prependString and abstract is unique")
    data = data.loc[
        :, ["id", "status", "prependString", "sourceAbstract", "evidenceLevel"]
    ]
    print("keeping columns:")
    print(["id", "status", "prependString", "sourceAbstract", "evidenceLevel"])
    return data


def get_stratified_train_test_split(data, test_size, class_col):
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[class_col]
    )
    return train_data, test_data


if __name__ == "__main__":
    create_folder_if_not_exists(os.path.join(PROJECT_ROOT, "data/02_processed"))

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/01_raw/civic_evidence.csv"))
    print(f"Retrieved {df.shape[0]} records")

    df = do_data_cleaning(df)

    print(f"Processed data set has {df.shape[0]} records")
    print(THE_CLASS_DISTRIBUTION_IS)
    print(df.evidenceLevel.value_counts(normalize=True))

    print("Creating stratified train-test split based on the full data")
    train_data_full, test_data_full = get_stratified_train_test_split(
        df, 0.2, "evidenceLevel"
    )
    print(f"The full training set has {train_data_full.shape[0]} records")
    print(THE_CLASS_DISTRIBUTION_IS)
    print(train_data_full.evidenceLevel.value_counts(normalize=True))
    print(f"The full test set has {test_data_full.shape[0]} records")
    print(THE_CLASS_DISTRIBUTION_IS)
    print(test_data_full.evidenceLevel.value_counts(normalize=True))

    print("Creating stratified train-test split based on accepted records only")
    train_data_accepted_only, test_data_accepted_only = get_stratified_train_test_split(
        df.loc[df.status == "ACCEPTED"], 0.2, "evidenceLevel"
    )
    print(
        f"The accepted only training set has {train_data_accepted_only.shape[0]} records"
    )
    print(THE_CLASS_DISTRIBUTION_IS)
    print(train_data_accepted_only.evidenceLevel.value_counts(normalize=True))
    print(f"The accepted only test set has {test_data_accepted_only.shape[0]} records")
    print(THE_CLASS_DISTRIBUTION_IS)
    print(test_data_accepted_only.evidenceLevel.value_counts(normalize=True))

    print("Success: Writing to CSV")
    train_data_full.to_csv(
        os.path.join(PROJECT_ROOT, "data/02_processed/civic_evidence_train.csv")
    )
    test_data_full.to_csv(
        os.path.join(PROJECT_ROOT, "data/02_processed/civic_evidence_test.csv")
    )
    train_data_accepted_only.to_csv(
        os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_train_accepted_only.csv"
        )
    )
    test_data_accepted_only.to_csv(
        os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_test_accepted_only.csv"
        )
    )
