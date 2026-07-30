#!/usr/bin/env python3
"""NBA 预测（ELO + 效率模型）"""

import json, os, sys, time, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from config import LEAGUES, ELO, PREDICT, TEAM_CN

# ODDS_API_KEY — 环境变量优先，配置文件兜底
ODDS_API_KEY = os.environ.get("ODDS_API_KEY") or ""
if not ODDS_API_KEY:
    try:
        from config import ODDS_API_KEY as _cfg_key
        ODDS_API_KEY = _cfg_key
    except ImportError:
        pass

# ── 路径 ──
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
ELO_FILE = DATA_DIR / "elo.json"
DATA_DIR.mkdir(exist_ok=True)

NOW = datetime.now(timezone.utc)
BJT = NOW + timedelta(hours=8)

# ── 工具 ──
def _get(url: str, headers: dict | None = None) -> list | dict:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def to_cn(name: str) -> str:
    return TEAM_CN.get(name, name)

# ── ELO ──
def load_elo() -> dict[str, float]:
    if ELO_FILE.exists():
        return json.loads(ELO_FILE.read_text())
    return {}

def save_elo(ratings: dict[str, float]):
    ELO_FILE.write_text(json.dumps(ratings, indent=2))

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(ratings: dict, home: str, away: str, home_win: bool):
    K = ELO["K"]
    ha = ELO["HOME_ADV"]
    rh = ratings.get(home, ELO["INITIAL"]) + ha
    ra = ratings.get(away, ELO["INITIAL"])
    eh = expected_score(rh, ra)
    sh = 1.0 if home_win else 0.0
    ratings[home] = ratings.get(home, ELO["INITIAL"]) + K * (sh - eh)
    ratings[away] = ratings.get(away, ELO["INITIAL"]) + K * ((1 - sh) - (1 - eh))

# ── 数据获取 ──
def fetch_odds_games() -> list[dict]:
    """从 The Odds API 获取 NBA 比赛 + 盘口"""
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set", file=sys.stderr)
        return []

    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?regions=us&markets=h2h,spreads,totals"
        f"&apiKey={ODDS_API_KEY}"
    )
    games = _get(url)
    return games

def fetch_team_stats() -> dict:
    """从 balldontlie.io 获取球队赛季场均数据"""
    stats = {}
    try:
        url = "https://api.balldontlie.io/v1/teams"
        teams = _get(url, {"Authorization": ""})
        # balldontlie 1.0 doesn't need auth for basic stats
        url2 = "https://api.balldontlie.io/v1/games?seasons[]=2025&per_page=100"
        games = _get(url2)
        # Parse team stats from games
        # 简化：先返回空，用 league average 兜底
        return stats
    except Exception as e:
        print(f"  [warn] fetch_team_stats failed: {e}", file=sys.stderr)
        return stats

# ── 预测 ──
def predict_game(home: str, away: str, ratings: dict,
                 odds_home: float | None, odds_away: float | None,
                 spread_line: float | None, total_line: float | None) -> dict:
    """单场预测"""
    rh = ratings.get(home, ELO["INITIAL"]) + ELO["HOME_ADV"]
    ra = ratings.get(away, ELO["INITIAL"])
    win_prob = expected_score(rh, ra)

    # 预测分差
    margin = (rh - ra) * 0.06 + PREDICT["HOME_SCORE_ADV"]  # 每 10 ELO 分 ≈ 0.6 分
    pred_home_score = PREDICT["AVG_PPG"] + margin / 2
    pred_away_score = PREDICT["AVG_PPG"] - margin / 2
    pred_total = pred_home_score + pred_away_score
    pred_margin = pred_home_score - pred_away_score

    # 方向判断
    direction = "主胜" if win_prob > 0.5 else "客胜"
    stars = "★" * min(5, int(abs(win_prob - 0.5) * 20))

    # 让分预测
    spread_pred = None
    if spread_line is not None:
        cover = "主" if pred_margin > spread_line else "客"
        spread_pred = f"{cover}(+{abs(pred_margin - spread_line):.1f})"

    # 大小分预测
    total_pred = None
    if total_line is not None:
        ou = "大" if pred_total > total_line else "小"
        total_pred = f"{ou}(+{abs(pred_total - total_line):.1f})"

    # 赔率校准
    odds_cal = ""
    if odds_home and odds_away:
        implied = 1 / odds_home + 1 / odds_away
        implied_home = (1 / odds_home) / implied
        edge = win_prob - implied_home
        odds_cal = f"赔率赢率={implied_home:.1%} 边缘={edge:+.1%}"

    return {
        "match": f"{to_cn(home)} vs {to_cn(away)}",
        "home": to_cn(home),
        "away": to_cn(away),
        "direction": direction,
        "stars": stars,
        "win_prob": round(win_prob, 3),
        "predicted_score": f"{pred_home_score:.0f}-{pred_away_score:.0f}",
        "predicted_margin": round(pred_margin, 1),
        "spread_prediction": spread_pred,
        "total_prediction": total_pred,
        "odds_calibration": odds_cal,
    }

# ── 主流程 ──
def run():
    print("=== NBA 预测 ===", file=sys.stderr)

    # 1. 加载 ELO
    ratings = load_elo()
    print(f"  ELO 球队数: {len(ratings)}", file=sys.stderr)

    # 2. 获取比赛
    games = fetch_odds_games()
    if not games:
        print("  [warn] 无比赛数据，返回空", file=sys.stderr)
        return {"league": "nba", "status": "no_data", "predictions": []}

    print(f"  比赛数: {len(games)}", file=sys.stderr)

    # 3. 预测
    predictions = []
    for g in games:
        home = g.get("home_team", "?")
        away = g.get("away_team", "?")
        bookmakers = g.get("bookmakers", [])
        odds_home = odds_away = spread_line = total_line = None

        for bm in bookmakers:
            for m in bm.get("markets", []):
                outcomes = m.get("outcomes", [])
                if m["key"] == "h2h" and len(outcomes) >= 2:
                    odds_home = outcomes[0]["price"] if outcomes[0]["name"] == home else outcomes[1]["price"]
                    odds_away = outcomes[1]["price"] if outcomes[0]["name"] == home else outcomes[0]["price"]
                    # 确保 odds_home 是主队
                    if outcomes[0]["name"] == home:
                        odds_home, odds_away = outcomes[0]["price"], outcomes[1]["price"]
                    else:
                        odds_home, odds_away = outcomes[1]["price"], outcomes[0]["price"]
                elif m["key"] == "spreads" and len(outcomes) >= 2:
                    # 让分线：主队有正 point 表示主队受让
                    for o in outcomes:
                        if o["name"] == home:
                            spread_line = o.get("point", 0)
                        elif o["name"] == away:
                            pass  # 客队让分是反的
                    # 简化：取第一个让分值
                    if outcomes[0].get("point") is not None:
                        spread_line = outcomes[0]["point"]
                elif m["key"] == "totals" and len(outcomes) >= 2:
                    total_line = outcomes[0].get("point", 0)

        pred = predict_game(home, away, ratings, odds_home, odds_away, spread_line, total_line)
        predictions.append(pred)

        # 更新 ELO（已结束比赛）
        if g.get("completed"):
            score = g.get("scores", {})
            hs = score.get(home, 0)
            as_ = score.get(away, 0)
            if hs and as_:
                update_elo(ratings, home, away, hs > as_)

    # 4. 保存 ELO
    save_elo(ratings)

    output = {
        "league": "nba",
        "generated_at": NOW.isoformat(),
        "status": "ok",
        "predictions": predictions,
        "total": len(predictions),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output

if __name__ == "__main__":
    run()