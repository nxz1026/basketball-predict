"""
NBA AI Feedback Loop — bridge between AI enrichment and prediction engine.

Flow:
  Day N:  predict.py → ai_enrich_gha.py → save_ai_scores()
  Day N+1: predict.py → load_ai_scores() → adjust prediction confidence → predict
  After match: reconcile_results() → track metrics

Usage:
  from ai.feedback_loop import load_ai_adjustments, save_ai_scores
  adjustments = load_ai_adjustments()
  # pass adjustments into predict_game()
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
AI_SCORES_FILE = BASE / "data" / "ai_scores.json"
METRICS_FILE = BASE / "data" / "metrics_history.json"


def load_ai_adjustments() -> dict[str, dict]:
    """Load AI enrichment scores from previous run."""
    if not AI_SCORES_FILE.exists():
        return {}
    try:
        with open(AI_SCORES_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_ai_scores(enriched_items: list[dict]) -> None:
    """Save AI enrichment scores for next prediction run."""
    existing = load_ai_adjustments()
    for item in enriched_items:
        name = item.get("name", "")
        if not name:
            continue
        notes = item.get("ai_notes", "")
        if "mock" in notes:
            continue
        existing[name] = {
            "ai_score": item.get("ai_score", 50),
            "ai_summary": item.get("ai_summary", ""),
            "ai_notes": notes,
        }

    AI_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AI_SCORES_FILE, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"[AI Feedback] Saved {len(existing)} AI scores to {AI_SCORES_FILE}")


def adjust_prediction(prediction: dict, ai_adjustments: dict[str, dict]) -> dict:
    """Apply AI enrichment scores to adjust prediction confidence.

    Formula:
      adjusted_confidence = base_confidence * (0.7 + 0.3 * ai_score/100)
    """
    match_name = prediction.get("match", "")
    adj = ai_adjustments.get(match_name)
    if not adj:
        return prediction

    ai_score = adj.get("ai_score", 50)
    base_conf = prediction.get("win_prob", 0.5)

    factor = 0.7 + 0.3 * (ai_score / 100)
    adjusted_conf = min(base_conf * factor, 1.0)

    prediction["win_prob"] = round(adjusted_conf, 3)
    prediction["ai_adjusted"] = True
    prediction["ai_score_used"] = ai_score
    prediction["ai_adjustment_factor"] = round(factor, 3)

    # Recalc stars
    confidence = abs(adjusted_conf - 0.5) * 2
    if confidence >= 0.5:
        prediction["stars"] = "3-star"
    elif confidence >= 0.3:
        prediction["stars"] = "2-star"
    else:
        prediction["stars"] = "1-star"

    # Recalc direction
    from config import TEAM_CN, TEAM_CN_ALIASES
    def to_cn(name):
        return TEAM_CN.get(name, TEAM_CN_ALIASES.get(name, name))

    home = prediction.get("home", "")
    away = prediction.get("away", "")
    winner = to_cn(home) if adjusted_conf > 0.5 else to_cn(away)
    prediction["direction"] = f"{winner} 胜"

    return prediction


def reconcile_results(predictions: list[dict], actual_results: list[dict]) -> dict:
    """Compare predictions against actual match results."""
    actual_by_name = {}
    for m in actual_results:
        name = m.get("match", "")
        score = m.get("predicted_score", "")
        if name and score:
            actual_by_name[name] = m

    correct = 0
    total = 0
    ai_correct = 0
    ai_total = 0
    details = []

    for p in predictions:
        match_name = p.get("match", "")
        actual = actual_by_name.get(match_name)
        if not actual:
            continue

        total += 1
        predicted_dir = p.get("direction", "")
        predicted_score = p.get("predicted_score", "0-0")

        try:
            home_goals = int(predicted_score.split("-")[0])
            away_goals = int(predicted_score.split("-")[1])
        except (ValueError, IndexError):
            continue

        if home_goals > away_goals:
            actual_dir = f"{p.get('home', '')} 胜"
        elif home_goals < away_goals:
            actual_dir = f"{p.get('away', '')} 胜"
        else:
            actual_dir = "平局"

        is_correct = predicted_dir == actual_dir or (
            "胜" in predicted_dir and "胜" in actual_dir
            and predicted_dir.split("胜")[0].strip() == actual_dir.split("胜")[0].strip()
        )

        if is_correct:
            correct += 1
            if p.get("ai_adjusted"):
                ai_correct += 1

        if p.get("ai_adjusted"):
            ai_total += 1

        details.append({
            "match": match_name,
            "predicted": predicted_dir,
            "actual": actual_dir,
            "correct": is_correct,
            "ai_adjusted": p.get("ai_adjusted", False),
            "ai_score": p.get("ai_score_used"),
            "confidence": p.get("win_prob"),
        })

    metrics = {
        "total_matches": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0,
        "ai_adjusted_total": ai_total,
        "ai_adjusted_correct": ai_correct,
        "ai_adjusted_accuracy": round(ai_correct / ai_total, 3) if ai_total else None,
        "details": details,
    }

    # Save metrics history
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                history = json.load(f)
        except (json.JSONDecodeError):
            pass

    from datetime import datetime, timezone
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    history.append(metrics)
    if len(history) > 100:
        history = history[-100:]

    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return metrics


def print_metrics_summary(metrics: dict) -> str:
    """Format metrics as human-readable string."""
    lines = [
        "=== AI Feedback Metrics ===",
        f"Matches: {metrics['total_matches']}",
        f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_matches']})",
    ]
    if metrics.get("ai_adjusted_total"):
        lines.append(
            f"AI-adjusted accuracy: {metrics['ai_adjusted_accuracy']:.1%} "
            f"({metrics['ai_adjusted_correct']}/{metrics['ai_adjusted_total']})"
        )
    return "\n".join(lines)