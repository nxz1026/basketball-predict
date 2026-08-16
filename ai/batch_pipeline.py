"""
Batch AI analysis pipeline: split items, enrich via Gemini, return scored items.
Adapted from ECC community data-scraper-agent skill.
"""
import json
from ai.llm_client import generate


def analyse_batch(
    items: list[dict],
    context: str = "",
    preference_prompt: str = "",
    config: dict = None,
) -> list[dict]:
    """Analyse items in batches. Returns items enriched with AI fields.

    Each item gets: ai_score (0-100), ai_summary, ai_notes.
    Items below min_score are filtered out.
    """
    config = config or {}
    ai_cfg = config.get("ai", {})
    model = ai_cfg.get("model", "gemini-2.5-flash")
    rate_limit = ai_cfg.get("rate_limit_seconds", 7.0)
    min_score = ai_cfg.get("min_score", 0)
    batch_size = ai_cfg.get("batch_size", 5)

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"  [AI] {len(items)} items → {len(batches)} API calls")

    enriched = []
    for i, batch in enumerate(batches):
        print(f"  [AI] Batch {i + 1}/{len(batches)}...")
        prompt = _build_prompt(batch, context, preference_prompt, config)
        result = generate(prompt, model=model, rate_limit=rate_limit)
        analyses = result.get("analyses", [])

        for j, item in enumerate(batch):
            ai = analyses[j] if j < len(analyses) else {}
            if ai:
                score = max(0, min(100, int(ai.get("score", 0))))
                if min_score and score < min_score:
                    continue
                enriched.append({
                    **item,
                    "ai_score": score,
                    "ai_summary": ai.get("summary", ""),
                    "ai_notes": ai.get("notes", ""),
                })
            else:
                enriched.append(item)

    return enriched


def _build_prompt(batch, context, preference_prompt, config):
    """Build the Gemini prompt for a batch of items."""
    priorities = config.get("priorities", [])
    items_text = "\n\n".join(
        f"Item {i+1}: {json.dumps({k: v for k, v in item.items() if not k.startswith('_')})}"
        for i, item in enumerate(batch)
    )

    return f"""Analyse these {len(batch)} items and return a JSON object.
# Items
{items_text}
# User Context
{context[:800] if context else "Not provided"}
# User Priorities
{chr(10).join(f"- {p}" for p in priorities)}
{preference_prompt}
# Instructions
Return: {{"analyses": [{{"score": <0-100>, "summary": "<2 sentences>", "notes": ""}} for each item in order]}}
Be concise. Score 90+=excellent match, 70-89=good, 50-69=ok, <50=weak."""