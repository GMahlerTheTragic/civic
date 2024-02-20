import json
from time import sleep
import ast

import torch
from openai import OpenAI
import pandas as pd

import os

from sklearn.metrics import f1_score

from civic.config import DATA_PROCESSED_DIR

client = OpenAI()

label_to_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def get_gpt_4_example(prepend_string, abstract):
    return f"""Metadata:\n\"{prepend_string}\"\nAbstract:\n:\"{abstract}\""""


def get_random_few_shot_examples(n_samples_per_evidence_level, is_test=False):
    if is_test:
        sampled_df = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4_mc.csv")
        )
    else:
        df = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train_mc.csv")
        )
        sampled_df = pd.DataFrame()
        for value in ["A", "B", "C", "D", "E"]:
            sampled_rows = df.loc[df[value] == 1].sample(n_samples_per_evidence_level)
            sampled_df = pd.concat([sampled_df, sampled_rows], axis=0)
    out = [
        {
            "example": sampled_df.iloc[i]["sourceAbstract"],
            "labels": sampled_df.iloc[i][["A", "B", "C", "D", "E"]].to_list(),
        }
        for i in range(sampled_df.shape[0])
    ]
    return out


def _get_prompt_from_sample(sample):
    return [
        {
            "role": "user",
            "content": f"Extract the levels of clinical evidence from this abstract: {sample['example']}",
        },
        {
            "role": "assistant",
            "content": f"""{sample['labels']}""",
        },
    ]


def gpt4_query(examples, prompt):
    messages = (
        [
            {
                "role": "system",
                "content": "You are an expert on rare tumor treatments. In the following you will be presented with "
                + "abstracts from medical research papers. These abstracts deal with treatment approaches for rare "
                + "cancers as characterized by their specific genomic variants. Your job is to infer matching levels of "
                + "clinical evidence of the investigations described in the abstract."
                + ' Labels range from "A" (indicating the strongest clinical evidence) to '
                + '"E" (indicating the weakest clinical evidence). A single abstract can be associated with multiple'
                + " evidence levels. The definitions of the evidence levels are as follows: "
                + " A: Proven/consensus association in human medicine."
                + " B: Clinical trial or other primary patient data supports association."
                + " C: Individual case reports from clinical journals."
                + " D: In vivo or in vitro models support association."
                + " E: Indirect evidence."
                + " You will answer machine-like with a list of 0 vs. 1 flags for every of the five"
                + " evidence levels indicating if the abstract matches an evidence level or not. In the following"
                + " you will be presented with examples. Note the first entry of the result list should correspond to evidence level "
                + '"A" and the last entry of the result list should correspond to evidence level "E".',
            },
            *examples,
            prompt,
        ],
    )
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview", messages=messages[0]
    )
    print(completion)
    projected_labels = ast.literal_eval(completion.choices[0].message.content)
    return projected_labels


def do_gpt4_evaluation(n_shots):
    train_examples = [
        _get_prompt_from_sample(sample)
        for sample in get_random_few_shot_examples(n_shots)
    ]
    examples = sum(train_examples, [])
    test_examples = [
        _get_prompt_from_sample(sample)
        for sample in get_random_few_shot_examples(5, is_test=True)
    ]
    predicted_labels = []
    actual_labels = []
    for i in range(len(test_examples)):
        if (i + 1) % 4 == 0:
            sleep(10)
        projected_label = gpt4_query(examples, test_examples[i][0])
        actual_label = ast.literal_eval(test_examples[i][1]["content"])

        predicted_labels.append(projected_label)
        actual_labels.append(actual_label)
        print(f"Projected label : {projected_label}")
        print(f"Actual label : {actual_label}")
    return {
        "f1-scores": f1_score(
            torch.tensor(actual_labels),
            torch.tensor(predicted_labels),
            average=None,
        ).tolist(),
        "f1-score": f1_score(
            torch.tensor(actual_labels),
            torch.tensor(predicted_labels),
            average="weighted",
        ),
    }


def main():
    metrics_dict = {}
    n_shots = 4
    print(f"using approx {n_shots * 700 * 5 * 25} tokens")
    metrics = do_gpt4_evaluation(n_shots)
    print(metrics)
    metrics_dict[str(n_shots)] = metrics
    with open(f"gpt4_results__nshots{n_shots}_2.json", "w") as json_file:
        json.dump(metrics_dict, json_file)


if __name__ == "__main__":
    main()
