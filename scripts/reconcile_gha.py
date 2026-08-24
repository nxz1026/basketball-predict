"""NBA GHA reconciliation — reconcile predictions vs results, append 命中率小结 to push.

Usage: python3 scripts/reconcile_gha.py
Reads /tmp/output.json; writes the accuracy summary into /tmp/email_body.txt.
"""
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_PATH = "/tmp/output.json"
EMAIL_BODY_PATH = "/tmp/email_body.txt"


def main():
    if not os.path.exists(OUTPUT_PATH):
        print("[Reconcile] No output data, skipping")
        return

    try:
        from ai.feedback_loop import (
            reconcile_results,
            print_metrics_summary,
            format_email_summary,
        )
    except ImportError as e:
        print(f"[Reconcile] import error: {e}")
        return

    with open(OUTPUT_PATH) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    past_details = data.get("past_games_detail", [])

    if not predictions or not past_details:
        print("[Reconcile] No predictions or past results to reconcile")
        return

    metrics = reconcile_results(predictions, past_details)
    print(print_metrics_summary(metrics))

    try:
        summary = format_email_summary(metrics)
        if os.path.exists(EMAIL_BODY_PATH):
            with open(EMAIL_BODY_PATH) as f:
                body = f.read()
            with open(EMAIL_BODY_PATH, "w") as f:
                f.write(body + summary + "\n")
            print("[Reconcile] Appended 命中率小结 to email body")
        else:
            print("[Reconcile] email body not found, skip append")
    except Exception as e:
        print(f"[Reconcile] Could not append summary: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Reconcile] 错误: {e}")
        print(traceback.format_exc())
        sys.exit(1)
