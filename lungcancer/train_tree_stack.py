from __future__ import annotations

import argparse
import json
import traceback

from .tree_stack import train_with_nested_cv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nested-CV tree stack for mortality/response")
    parser.add_argument("--task", choices=["mortality", "response", "both"], default="both")
    parser.add_argument("--algorithm", choices=["lightgbm", "xgboost", "tabnet", "all"], default="all")
    parser.add_argument(
        "--max-rows-mortality",
        type=int,
        default=None,
        help="Optional cap on mortality rows (stratified sample) to reduce runtime.",
    )
    args = parser.parse_args()

    tasks = ["mortality", "response"] if args.task == "both" else [args.task]
    algorithms = ["lightgbm", "xgboost", "tabnet"] if args.algorithm == "all" else [args.algorithm]

    output = []
    for task in tasks:
        for algo in algorithms:
            print(f"[train] task={task} algorithm={algo}", flush=True)
            try:
                summary = train_with_nested_cv(
                    task=task,
                    algorithm=algo,
                    max_rows=(args.max_rows_mortality if task == "mortality" else None),
                )
                output.append(summary)
                print(f"[done] task={task} algorithm={algo}", flush=True)
            except Exception as exc:
                output.append(
                    {
                        "task": task,
                        "algorithm": algo,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[error] task={task} algorithm={algo}: {exc}", flush=True)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
