"""NBA GHA reconciliation — reconcile predictions vs results, append 命中率小结 to push.

Usage: python3 scripts/reconcile_gha.py
Reads /tmp/output.json; writes the accuracy summary into /tmp/email_body.txt.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
OUTPUT_PATH = "/tmp/output.json"
EMAIL_BODY_PATH = "/tmp/email_body.txt"
SAVED_PRED_PATH = BASE / "data" / "predictions_history.json"


def load_saved_predictions() -> list[dict]:
    if not SAVED_PRED_PATH.exists():
        return []
    try:
        return json.loads(SAVED_PRED_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_saved_predictions(preds: list[dict]) -> None:
    SAVED_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)


def _is_stale(p: dict, max_age_days: int = 90) -> bool:
    ct = p.get("commence_time", "")
    if not ct:
        return False
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days > max_age_days
    except ValueError:
        return False


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

    past_details = data.get("past_games_detail", [])

    # 用跨运行保存的历史预测与今日完赛结果核对（同日运行的 predictions 只含未来场，无法自比）
    saved = load_saved_predictions()
    by_match = {}
    for p in saved:
        by_match[p.get("match")] = p  # 同名比赛保留最新一条
    saved = list(by_match.values())

    if not saved or not past_details:
        print("[Reconcile] No predictions or past results to reconcile")
        return

    metrics = reconcile_results(saved, past_details)
    print(print_metrics_summary(metrics))

    # 已核对的预测从历史库移除，避免重复计数；并清理过期条目
    reconciled = {d.get("match") for d in metrics.get("details", [])}
    remaining = [
        p for p in load_saved_predictions()
        if p.get("match") not in reconciled and not _is_stale(p)
    ]
    save_saved_predictions(remaining)

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
