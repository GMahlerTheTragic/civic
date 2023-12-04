from openai import OpenAI
import pandas as pd

import os

from civic.config import DATA_PROCESSED_DIR

client = OpenAI()


def get_gpt_4_example(prependString, abstract):
    return f"""METADATA: \"{prependString}\"\n ABSTRACT: \"{abstract}\""""


def get_random_samples(n_samples_per_evidence_level, isVal=False):
    if isVal:
        df_1 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv"))
        df_2 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv"))
        weights = 1 / df_2.evidenceLevel.value_counts()
        weights = weights / weights.sum()
        print(weights)
        df = pd.concat([df_1, df_2], axis=0).reset_index()
    else:
        df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv"))
    sampled_df = pd.DataFrame()
    for value in df["evidenceLevel"].unique():
        sampled_rows = df[df["evidenceLevel"] == value].sample(
            n_samples_per_evidence_level
        )
        sampled_df = pd.concat([sampled_df, sampled_rows], axis=0)
    out = [
        {
            "example": sampled_df.iloc[i]["prependString"]
            + "\n"
            + sampled_df.iloc[i]["sourceAbstract"],
            "label": sampled_df.iloc[i]["evidenceLevel"],
        }
        for i in range(sampled_df.shape[0])
    ]
    return out


def _get_prompt_from_sample(sample):
    return [
        {
            "role": "user",
            "content": f"Extract the level of clinical significance from this combination of metadata and abstract:\n{sample['example']}",
        },
        {
            "role": "assistant",
            "content": f"""{sample['label']}""",
        },
    ]


def main():
    examples = [_get_prompt_from_sample(sample) for sample in get_random_samples(1)]
    examples = sum(examples, [])
    prompts = [
        _get_prompt_from_sample(sample) for sample in get_random_samples(1, isVal=True)
    ]
    print(examples[1])
    for i in range(len(prompts)):
        messages = (
            [
                {
                    "role": "system",
                    "content": """You are an expert on rare tumor treatments. In the following you will be presented with
                        abstracts from medical research papers. These abstracts deal with treatment approaches for rare
                        cancers as characterized by their specific genomic variants. Your job is to infer the level of
                        clinical significance of the investigations described in the abstract from the abstract and relevant
                        metadata. The labels should range from \"A\" (indicating the strongest clinical significance) to
                        \"E\" (indicating the weakest clinical significance). You will answer machine-like with exactly one
                        character (the level of clinical significance you think is appropriate). You will be presented with
                        examples.""",
                },
                *examples,
                prompts[i][0],
            ],
        )
        # completion = client.chat.completions.create(
        #     model="gpt-4-1106-preview", messages=messages[0]
        # )
        # projected_label = completion.choices[0].message
        # actual_label = prompts[i][1]["content"]
        # print(f"Projected label : {projected_label}")
        # print(f"Actual label : {actual_label}")


if __name__ == "__main__":
    main()
