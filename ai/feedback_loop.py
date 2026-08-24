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
    with open(AI_SCORES_FILE, "w", encoding="utf-8") as f:
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
    """Compare predictions against actual match results.

    `actual_results` entries (past_games_detail) use keys: name, home, away, score.
    Predictions use keys: match, home, away, direction, win_prob, ai_adjusted.
    """
    from config import TEAM_CN, TEAM_CN_ALIASES

    def to_cn(name: str) -> str:
        return TEAM_CN.get(name, TEAM_CN_ALIASES.get(name, name))

    actual_by_name = {}
    for m in actual_results:
        name = m.get("match") or m.get("name", "")
        score = m.get("score") or m.get("predicted_score", "")
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
        predicted_winner = predicted_dir.replace("胜", "").strip() if "胜" in predicted_dir else ""

        score = actual.get("score") or actual.get("predicted_score", "")
        try:
            hs, as_ = str(score).split("-")
            home_goals = int(hs)
            away_goals = int(as_)
        except (ValueError, AttributeError, TypeError):
            continue

        home_cn = to_cn(p.get("home", ""))
        away_cn = to_cn(p.get("away", ""))
        if home_goals > away_goals:
            actual_winner = home_cn
        elif away_goals > home_goals:
            actual_winner = away_cn
        else:
            actual_winner = ""

        is_correct = bool(predicted_winner) and predicted_winner == actual_winner

        if is_correct:
            correct += 1
            if p.get("ai_adjusted"):
                ai_correct += 1
        if p.get("ai_adjusted"):
            ai_total += 1

        details.append({
            "match": match_name,
            "predicted": predicted_dir,
            "actual": (actual_winner + " 胜") if actual_winner else "未知",
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

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return metrics


def compute_rolling_accuracy(history: list[dict], window: int = 30) -> dict:
    """Aggregate accuracy across the most recent `window` reconciliation runs."""
    recent = history[-window:] if window else history
    total = sum(m.get("total_matches", 0) for m in recent)
    correct = sum(m.get("correct", 0) for m in recent)
    ai_total = sum(m.get("ai_adjusted_total", 0) for m in recent)
    ai_correct = sum(m.get("ai_adjusted_correct", 0) for m in recent)
    return {
        "window": window,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0,
        "ai_total": ai_total,
        "ai_correct": ai_correct,
        "ai_accuracy": round(ai_correct / ai_total, 3) if ai_total else None,
    }


def format_email_summary(metrics: dict) -> str:
    """Build a Markdown-ish 命中率小结 block for the daily push."""
    lines = ["", "=== 命中率小结 ===", ""]
    if metrics.get("total_matches"):
        lines.append(
            f"本日命中率: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_matches']})"
        )
        if metrics.get("ai_adjusted_total"):
            lines.append(
                f"AI 调整命中率: {metrics['ai_adjusted_accuracy']:.1%} "
                f"({metrics['ai_adjusted_correct']}/{metrics['ai_adjusted_total']})"
            )
    else:
        lines.append("本日无回填比赛可供核对")

    # Rolling accuracy from saved history (includes today's run)
    try:
        history = json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else []
    except (json.JSONDecodeError, OSError):
        history = []
    if history:
        roll = compute_rolling_accuracy(history, window=30)
        if roll["total"]:
            lines.append(
                f"近30次累计命中率: {roll['accuracy']:.1%} ({roll['correct']}/{roll['total']})"
            )
            if roll["ai_total"]:
                lines.append(
                    f"近30次AI调整命中率: {roll['ai_accuracy']:.1%} "
                    f"({roll['ai_correct']}/{roll['ai_total']})"
                )

    return "\n".join(lines)


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