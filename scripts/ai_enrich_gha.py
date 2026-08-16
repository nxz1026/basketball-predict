"""
AI Enrichment for NBA GHA — reads prediction output, enriches via LLM, appends to email body.
Usage: python3 scripts/ai_enrich_gha.py
Env: LLM_API_KEY, LLM_MODEL (optional — skip if none set)
"""
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def load_predictions(path: str = "/tmp/output.json") -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    predictions = data.get("predictions", [])
    return predictions if isinstance(predictions, list) else []


def enrich_via_llm(predictions: list) -> tuple[str, list]:
    from ai.feedback_loop import save_ai_scores

    # Build items from predictions
    items = []
    for p in predictions[:10]:  # top 10
        items.append({
            "name": p.get("match", "?"),
            "source": "NBA",
            "url": "",
            "date_found": "",
            "direction": p.get("direction", "?"),
            "stars": p.get("stars", "?"),
            "confidence": str(p.get("win_prob", "")),
        })

    if not items:
        return "", []

    # Use the same batch pipeline if available, otherwise inline
    try:
        from ai.batch_pipeline import analyse_batch
    except ImportError:
        print("[AI Enrich] ai.batch_pipeline not available, using inline enrichment")
        return "", []

    enriched = analyse_batch(
        items,
        context="NBA basketball match predictions. Score each match by predicted value and confidence.",
        preference_prompt="",
        config={
            "ai": {
                "model": os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
                "batch_size": 5,
                "rate_limit_seconds": 3,
                "min_score": 0,
            },
            "priorities": [
                "High confidence predictions preferred",
                "Underdog picks preferred",
                "Clear direction signals preferred",
            ],
        },
    )

    lines = ["", "=== AI 分析 ===", ""]
    for item in enriched:
        s = item.get("ai_score", 0)
        summary = item.get("ai_summary", "")
        notes = item.get("ai_notes", "")
        if "mock" in notes:
            return "", []
        lines.append(f"• {item['name']}  **{s}/100**  — {summary}")
    lines.append("")
    return "\n".join(lines), enriched


def main():
    predictions = load_predictions()
    if not predictions:
        print("[AI Enrich] No prediction data found, skipping")
        return

    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("[AI Enrich] No LLM_API_KEY set, skipping enrichment")
        return

    print(f"[AI Enrich] Loaded {len(predictions)} predictions, enriching...")
    snippet, enriched_items = enrich_via_llm(predictions)
    if not snippet:
        print("[AI Enrich] No enrichment produced (mock or empty)")
        return

    # Save AI scores for next prediction run
    if enriched_items:
        from ai.feedback_loop import save_ai_scores
        save_ai_scores(enriched_items)

    email_body_path = "/tmp/email_body.txt"
    if os.path.exists(email_body_path):
        with open(email_body_path) as f:
            existing = f.read()
        with open(email_body_path, "w") as f:
            f.write(existing + snippet)
        print(f"[AI Enrich] Appended enrichment to {email_body_path}")
    else:
        with open(email_body_path, "w") as f:
            f.write(snippet)
        print(f"[AI Enrich] Created {email_body_path} with enrichment")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[AI Enrich] ❌ 错误: {e}")
        print(f"[AI Enrich] 详细: {traceback.format_exc()}")
        print(f"[AI Enrich] 提示: 检查依赖安装或 LLM_API_KEY 配置")
        sys.exit(1)