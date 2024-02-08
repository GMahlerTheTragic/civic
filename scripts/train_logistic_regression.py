from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import BertTokenizer
import os


from civic.utils.filesystem_utils import create_folder_if_not_exists
from civic.config import DATA_PROCESSED_DIR, DATA_RESULTS_DIR

t = BertTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)
vocabulary = t.get_vocab()
vocabulary_tokens = list(vocabulary.keys())


def _custom_analyzer(doc):
    tokens = t.tokenize(doc)
    bigrams = [" ".join(b) for b in zip(tokens[:-1], tokens[1:])]
    return bigrams


def _prepare_inputs():
    df_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv"))
    df_val = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv"))
    df_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv"))
    df_concat = pd.concat([df_train, df_val, df_test])
    df_concat["input"] = df_concat["prependString"] + df_concat["sourceAbstract"]
    corpus_concat = df_concat["input"].tolist()
    vectorizer = TfidfVectorizer(analyzer=_custom_analyzer)
    x = vectorizer.fit_transform(corpus_concat)
    y = df_concat["evidenceLevel"].tolist()
    start_val = df_train.shape[0]
    start_test = start_val + df_val.shape[0]
    x_cv = x[:start_test]
    y_cv = y[:start_test]
    x_train = x[:start_val]
    y_train = y[:start_val]
    x_test = x[start_test:]
    y_test = y[start_test:]
    return x_train, y_train, x_test, y_test, x_cv, y_cv


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_cv, y_cv = _prepare_inputs()
    logistic_regression = LogisticRegression(max_iter=1000, multi_class="multinomial")
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
    }
    grid_search = GridSearchCV(
        estimator=logistic_regression, param_grid=param_grid, cv=5, n_jobs=-1
    )

    grid_search.fit(x_cv, y_cv)

    print("Best Hyperparameters:", grid_search.best_params_)

    best_model = LogisticRegression(
        max_iter=1000, multi_class="multinomial", **grid_search.best_params_
    )
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred, average=None)
    micro_f1 = f1_score(y_test, y_pred, average="weighted")
    print("Accuracy", accuracy)
    print("Micro F1 Score:", micro_f1)
    print("F1 Scores:", f1s)

    a = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=best_model.classes_,
        normalize="true",
        cmap="Blues",
        values_format=".3f",
    )
    save_folder = os.path.join(DATA_PROCESSED_DIR, "logistic_regression")
    create_folder_if_not_exists(DATA_RESULTS_DIR)

    a.figure_.savefig(
        os.path.join(save_folder, "logistic_regression_confusion_matrix.pdf")
    )
