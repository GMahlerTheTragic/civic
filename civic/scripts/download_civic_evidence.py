import os

import pandas as pd
import requests

from civic.utils.filesystem_utils import create_folder_if_not_exists
from config import PROJECT_ROOT

CIVIC_GRAPH_QL_ENDPOINT = "https://civicdb.org/api/graphql"

GRAPH_QL_QUERY = """
    {{
        evidenceItems(first: {}, after: {}) {{
            totalCount
            pageCount
            nodes {{
                id
                status
                molecularProfile {{
                    id
                    name
                }}
                evidenceType
                evidenceLevel
                evidenceRating
                evidenceDirection
                description
                disease {{
                    id
                    name
                }}
                therapies {{
                    id
                    name
                }}
                source {{
                    abstract
                    id
                }}
                variantOrigin
                significance
            }}
            pageInfo {{
                endCursor
                hasNextPage
            }}
        }}
    }}
"""


def download_civic_evidence(first: int, after: str):
    response = requests.post(
        CIVIC_GRAPH_QL_ENDPOINT, json={"query": GRAPH_QL_QUERY.format(first, after)}
    )
    return response.json().get("data")


def clean_columns(json):
    _expand_column(
        json,
        "molecularProfile",
        {"id": "molecularProfileId", "name": "molecularProfileName"},
    )
    _expand_column(json, "disease", {"id": "diseaseId", "name": "diseaseName"})
    _expand_column(json, "source", {"abstract": "sourceAbstract", "id": "sourceId"})
    _expand_column(
        json,
        "therapies",
        {"id": "therapyIds", "name": "therapyNames"},
        is_list=True,
    )
    return json


def _expand_column(json, name, replacements, is_list=False):
    if json[name] is not None:
        for k, v in replacements.items():
            if is_list:
                json[v] = list(map(lambda x: x[k], json[name]))
            else:
                json[v] = json[name][k]
    json.pop(name)


if __name__ == "__main__":
    create_folder_if_not_exists(os.path.join(PROJECT_ROOT, "data/01_raw"))
    items = []
    print(f"Starting download from {CIVIC_GRAPH_QL_ENDPOINT}")
    data = download_civic_evidence(50, '""')
    items.extend(data["evidenceItems"]["nodes"])
    page = 1
    page_count = data["evidenceItems"]["pageCount"]
    print(f"\rDownloaded {page}/{page_count} pages", end="", flush=True)
    while data["evidenceItems"]["pageInfo"]["hasNextPage"]:
        page += 1
        data = download_civic_evidence(
            50, f'"{data["evidenceItems"]["pageInfo"]["endCursor"]}"'
        )
        items.extend(data["evidenceItems"]["nodes"])
        print(f"\rDownloaded {page}/{page_count} pages", end="", flush=True)
    print(f"\nFinished download from {CIVIC_GRAPH_QL_ENDPOINT}")
    print("Converting to pandas dataframe and expanding columns")
    df = pd.DataFrame(list(map(clean_columns, items)))
    df = df.sort_index(axis=1)
    total_count_per_database = data["evidenceItems"]["totalCount"]
    print(
        f"Retrieved {df.id.nunique()} unique evidence items. Database indicates {total_count_per_database}"
    )
    print(f"Saving as CSV file")
    df.to_csv(os.path.join(PROJECT_ROOT, "data/01_raw/civic_evidence.csv"))
    print(f"Success!")
