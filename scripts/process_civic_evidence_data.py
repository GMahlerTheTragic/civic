import sys
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from civic.utils.filesystem_utils import create_folder_if_not_exists
from civic.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

sys.path.append("..")


THE_CLASS_DISTRIBUTION_IS = "The class distribution is"

KEEP_COLS = [
    "id",
    "status",
    "prependString",
    "sourceAbstract",
    "sourceId",
    "evidenceLevel",
]


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

    print("Compiling prepend string")
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
        :,
        [
            "id",
            "status",
            "prependString",
            "sourceAbstract",
            "sourceId",
            "evidenceLevel",
        ],
    ]
    return data


def _compile_multi_class_data_set(data):
    print("Compiling multi class dataset")
    data_multi_class = (
        data.pivot_table(
            columns="evidenceLevel",
            index="sourceId",
            values="id",
            aggfunc=lambda x: 1 if len(x) >= 1 else 0,
        )
        .fillna(0)
        .reset_index()
    )
    data_multi_class = data_multi_class.merge(
        data[["sourceId", "sourceAbstract"]].drop_duplicates(),
        on="sourceId",
        how="left",
    )
    assert data_multi_class["sourceAbstract"].isna().sum() == 0
    return data_multi_class


def filter_for_unique_abstracts(data):
    return data.drop_duplicates(["sourceId"], keep=False).reset_index()


def get_stratified_train_test_split(data, test_size, class_col):
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=data[class_col], random_state=42
    )
    return train_data, test_data


if __name__ == "__main__":
    create_folder_if_not_exists(DATA_PROCESSED_DIR)

    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "civic_evidence.csv"))
    print(f"Retrieved {df.shape[0]} records")

    df = do_data_cleaning(df)
    df_unique_abstracts = filter_for_unique_abstracts(df)
    df_multi_class = _compile_multi_class_data_set(df)

    print(f"Processed data set has {df.shape[0]} records")
    print(f"Unique abstracts data set has {df_unique_abstracts.shape[0]} records")
    print(f"Multi class data set has {df_multi_class.shape[0]} records")

    print("Creating stratified train-test split based on the full data")
    train_data_full, test_data_full = get_stratified_train_test_split(
        df, 0.2, "evidenceLevel"
    )
    val_data_full, test_data_full = get_stratified_train_test_split(
        test_data_full, 0.5, "evidenceLevel"
    )
    train_data_unique_abstracts, test_data_unique_abstracts = (
        get_stratified_train_test_split(df_unique_abstracts, 0.2, "evidenceLevel")
    )
    val_data_unique_abstracts, test_data_unique_abstracts = (
        get_stratified_train_test_split(
            test_data_unique_abstracts, 0.5, "evidenceLevel"
        )
    )
    train_data_multi_class, test_data_multi_class = get_stratified_train_test_split(
        df_multi_class, 0.2, ["B", "C", "D"]
    )
    val_data_multi_class, test_data_multi_class = get_stratified_train_test_split(
        test_data_multi_class, 0.5, ["B", "C", "D"]
    )
    print(f"The full training set has {train_data_full.shape[0]} records")
    print(
        f"The unique abstracts training set has {train_data_unique_abstracts.shape[0]} records"
    )

    print(f"The full val set has {val_data_full.shape[0]} records")
    print(
        f"The unique abstracts val set has {val_data_unique_abstracts.shape[0]} records"
    )

    print(f"The full test set has {test_data_full.shape[0]} records")
    print(
        f"The unique abstracts test set has {test_data_unique_abstracts.shape[0]} records"
    )

    print("Creating Test Data Set for GPT-4 API - using 5 rows per evidence Level.")
    sampled_rows = [
        test_data_full[test_data_full["evidenceLevel"] == i].sample(5)
        for i in ["A", "B", "C", "D", "E"]
    ]
    test_data_full_gpt_4 = pd.concat(sampled_rows, axis=0).reset_index(drop=True)
    print(
        "Creating Test Data Set for GPT-4 API (unique abstracts) - using 3 rows per evidence Level."
    )
    sampled_rows = [
        test_data_unique_abstracts[
            test_data_unique_abstracts["evidenceLevel"] == i
        ].sample(3)
        for i in ["A", "B", "C", "D", "E"]
    ]
    test_data_unique_abstracts_gpt_4 = pd.concat(sampled_rows, axis=0).reset_index(
        drop=True
    )
    print("Creating Test Data Set for GPT-4 API - using 4 rows per evidence Level.")
    sampled_rows = [
        test_data_multi_class[test_data_multi_class[i] == 1].sample(4)
        for i in ["A", "B", "C", "D", "E"]
    ]
    test_data_multi_class_gpt_4 = pd.concat(sampled_rows, axis=0).reset_index(drop=True)
    assert test_data_multi_class_gpt_4.duplicated().sum() == 0

    print("Success: Writing to CSV")
    train_data_full.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    val_data_full.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    test_data_full.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    test_data_full_gpt_4.to_csv(
        os.path.join(
            DATA_PROCESSED_DIR,
            "civic_evidence_test_gpt4.csv",
        ),
        index=False,
        columns=KEEP_COLS,
    )
    train_data_unique_abstracts.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train_ua.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    val_data_unique_abstracts.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val_ua.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    test_data_unique_abstracts.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_ua.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    test_data_unique_abstracts_gpt_4.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4_ua.csv"),
        index=False,
        columns=KEEP_COLS,
    )
    train_data_multi_class.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train_mc.csv"),
        index=False,
        columns=["sourceAbstract", "A", "B", "C", "D", "E"],
    )
    val_data_multi_class.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val_mc.csv"),
        index=False,
        columns=["sourceAbstract", "A", "B", "C", "D", "E"],
    )
    test_data_multi_class.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_mc.csv"),
        index=False,
        columns=["sourceAbstract", "A", "B", "C", "D", "E"],
    )
    test_data_multi_class_gpt_4.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4_mc.csv"),
        index=False,
        columns=["sourceAbstract", "A", "B", "C", "D", "E"],
    )
