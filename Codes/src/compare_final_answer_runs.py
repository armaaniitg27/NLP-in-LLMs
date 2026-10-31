import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_FILE = RESULTS / "three_method_answer_comparison_28q.csv"

FILES = {
    "evidence_constrained": RESULTS / "evidence_constrained_results_28q.csv",
    "query_rewriting": RESULTS / "query_rewriting_results_28q.csv",
    "reranked": RESULTS / "reranked_answer_generation_results_28q.csv",
}

KEYS = ["question_id", "variant"]
FIELDS = [
    "original_question",
    "query_used",
    "reference_answer",
    "generated_answer",
    "retrieved_titles",
    "retrieved_context",
]


def main():
    frames = []

    for method, path in FILES.items():
        df = pd.read_csv(path)
        df["method"] = method

        keep = KEYS + ["method"] + [
            column for column in FIELDS if column in df.columns
        ]
        frames.append(df[keep])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["question_id", "variant", "method"],
        kind="stable",
    )

    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
