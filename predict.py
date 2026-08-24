#!/usr/bin/env python3
"""NBA 预测（ELO + 赔率模型）— 24小时窗口，参考 football-predict 流程"""

import json, os, sys, time, urllib.request, urllib.error, logging, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from config import LEAGUES, ELO, PREDICT, TEAM_CN, TEAM_CN_ALIASES

# ── 日志 ──
logger = logging.getLogger("nba_predict")

# ── AI 反馈循环 ──
try:
    from ai.feedback_loop import load_ai_adjustments, adjust_prediction
    _AI_ADJUSTMENTS = load_ai_adjustments()
    if _AI_ADJUSTMENTS:
        logger.info("AI 调整: 已加载 %d 条历史评分", len(_AI_ADJUSTMENTS))
    else:
        _AI_ADJUSTMENTS = {}
except Exception:
    _AI_ADJUSTMENTS = {}

# ── ODDS_API_KEY ──
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
BJT = NOW.astimezone(timezone(timedelta(hours=8)))
WINDOW_END = BJT + timedelta(hours=24)  # 24小时预测窗口

# ── 工具 ──
def _get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[list | dict | None, dict]:
    """HTTP GET 请求，返回 (数据, 响应头) 元组"""
    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            resp_headers = dict(r.headers)
            return data, resp_headers
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        safe_url = url.split("?")[0] + "?***"
        logger.error("request failed (%s): %s", safe_url, e)
        return None, {}

def to_cn(name: str) -> str:
    return TEAM_CN.get(name, TEAM_CN_ALIASES.get(name, name))

# ── ELO ──
def load_elo() -> dict[str, float]:
    if ELO_FILE.exists():
        return json.loads(ELO_FILE.read_text())
    return {team: ELO["INITIAL"] for team in TEAM_CN.keys()}

def save_elo(ratings: dict[str, float]) -> None:
    ELO_FILE.write_text(json.dumps(ratings, indent=2))

def apply_regression(ratings: dict[str, float]) -> int:
    """休赛期 ELO 回归：各队评分向初始分靠拢"""
    init = ELO["INITIAL"]
    reg = ELO["REGRESSION"]
    for team in list(ratings.keys()):
        ratings[team] = init + reg * (ratings[team] - init)
    return len(ratings)

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / ELO["SCALE"]))

def update_elo(ratings: dict, home: str, away: str, home_win: bool) -> None:
    K = ELO["K"]
    ha = ELO["HOME_ADV"]
    rh = ratings.get(home, ELO["INITIAL"]) + ha
    ra = ratings.get(away, ELO["INITIAL"])
    eh = expected_score(rh, ra)
    sh = 1.0 if home_win else 0.0
    ratings[home] = ratings.get(home, ELO["INITIAL"]) + K * (sh - eh)
    ratings[away] = ratings.get(away, ELO["INITIAL"]) + K * ((1 - sh) - (1 - eh))

def update_elo_from_results(ratings: dict[str, float], completed_games: list[dict]) -> int:
    """从已完赛比赛更新 ELO 评分"""
    updated = 0
    for g in completed_games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        scores = g.get("scores", {})
        if not isinstance(scores, dict):
            continue
        hs = scores.get(home, 0)
        as_ = scores.get(away, 0)
        if home and away and hs and as_:
            update_elo(ratings, home, away, hs > as_)
            updated += 1
    return updated

# ── 数据获取 ──
def fetch_odds_games(league: str = "nba") -> list[dict]:
    """从 The Odds API 获取比赛 + 盘口，去重后返回"""
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set")
        return []

    sport = LEAGUES.get(league, {}).get("odds_sport", "basketball_nba")
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?regions=us&markets=h2h,spreads,totals"
        f"&apiKey={ODDS_API_KEY}"
    )
    games, resp_headers = _get(url)
    if not isinstance(games, list):
        logger.error("unexpected API response type: %s", type(games))
        return []

    # 配额监控
    remaining = resp_headers.get("x-requests-remaining")
    if remaining is not None:
        logger.info("API 配额剩余: %s", remaining)
        try:
            if int(remaining) < 50:
                logger.warning("API 配额即将耗尽，剩余 %s 次", remaining)
        except ValueError:
            pass

    # 去重：按队伍组合
    seen_pairs = set()
    unique_games = []
    for g in games:
        h, a = g.get("home_team", ""), g.get("away_team", "")
        pair_key = tuple(sorted([h, a]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_games.append(g)

    if len(unique_games) < len(games):
        logger.info("API 返回 %d 条，去重后 %d 场", len(games), len(unique_games))

    return unique_games

def fetch_completed_games(league: str = "nba") -> list[dict]:
    """从 The Odds API 获取已完赛结果"""
    if not ODDS_API_KEY:
        return []

    sport = LEAGUES.get(league, {}).get("odds_sport", "basketball_nba")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/?apiKey={ODDS_API_KEY}"
    data, _ = _get(url)
    if not isinstance(data, list):
        return []
    return [g for g in data if g.get("completed") and g.get("scores")]

def filter_24h_games(games: list[dict]) -> tuple[list[dict], list[dict]]:
    """按24小时窗口过滤比赛，返回 (past, future)"""
    past = []
    future = []

    for g in games:
        comm_str = g.get("commence_time", "")
        if not comm_str:
            continue
        try:
            comm_dt = datetime.fromisoformat(comm_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        # 已结束或24小时内结束
        if comm_dt < NOW:
            past.append(g)
        elif comm_dt <= WINDOW_END:
            future.append(g)
        # 超过24小时的跳过

    logger.info(f"24h窗口: 已结束 {len(past)} 场，待预测 {len(future)} 场")
    return past, future

# ── 盘口解析 ──
def parse_odds(game: dict, home: str) -> tuple[float | None, float | None, float | None, float | None]:
    """解析赔率、让分线、大小分线"""
    bookmakers = game.get("bookmakers", [])
    odds_home = odds_away = spread_line = total_line = None

    for bm in bookmakers:
        for m in bm.get("markets", []):
            outcomes = m.get("outcomes", [])
            key = m.get("key", "")
            if key == "h2h" and len(outcomes) >= 2:
                if outcomes[0].get("name") == home:
                    odds_home, odds_away = outcomes[0].get("price"), outcomes[1].get("price")
                else:
                    odds_home, odds_away = outcomes[1].get("price"), outcomes[0].get("price")
            elif key == "spreads" and len(outcomes) >= 2:
                for o in outcomes:
                    if o.get("name") == home:
                        spread_line = o.get("point")
            elif key == "totals" and len(outcomes) >= 2:
                total_line = outcomes[0].get("point", 0)

    return odds_home, odds_away, spread_line, total_line

# ── 预测 ──
def predict_game(home: str, away: str, ratings: dict,
                 odds_home: float | None, odds_away: float | None,
                 spread_line: float | None, total_line: float | None) -> dict:
    """单场预测"""
    rh = ratings.get(home, ELO["INITIAL"]) + ELO["HOME_ADV"]
    ra = ratings.get(away, ELO["INITIAL"])
    elo_win_prob = expected_score(rh, ra)

    # 赔率反推胜率
    prob_home_from_odds = None
    if odds_home and odds_away and odds_home > 0 and odds_away > 0:
        imp_h = 1 / odds_home
        imp_a = 1 / odds_away
        total_imp = imp_h + imp_a
        if total_imp > 0:
            prob_home_from_odds = imp_h / total_imp

    # 融合 ELO 和赔率胜率
    if prob_home_from_odds is not None:
        win_prob = 0.3 * elo_win_prob + 0.7 * prob_home_from_odds
    else:
        win_prob = elo_win_prob

    # 预测分差 - 用赔率胜率推算
    if prob_home_from_odds and 0 < prob_home_from_odds < 1:
        logit = math.log(prob_home_from_odds / (1 - prob_home_from_odds))
        pred_margin = max(-20, min(20, logit * 14))
    else:
        pred_margin = (rh - ra) * PREDICT["ELO_TO_POINTS"] + PREDICT["HOME_SCORE_ADV"]

    pred_home_score = PREDICT["AVG_PPG"] + pred_margin / 2
    pred_away_score = PREDICT["AVG_PPG"] - pred_margin / 2
    pred_total = pred_home_score + pred_away_score

    # 方向判断
    winner = to_cn(home) if win_prob > 0.5 else to_cn(away)
    direction = f"{winner} 胜"
    confidence = abs(win_prob - 0.5) * 2
    if confidence >= 0.5:
        stars = "3-star"
    elif confidence >= 0.3:
        stars = "2-star"
    else:
        stars = "1-star"

    # 让分预测
    spread_pred = None
    if spread_line is not None and confidence >= PREDICT["SPREAD_CONFIDENCE"]:
        cover = "主" if pred_margin > -spread_line else "客"
        spread_pred = f"{cover}({pred_margin + spread_line:+.1f})"

    # 大小分预测
    total_pred = None
    if total_line is not None and confidence >= PREDICT["TOTAL_CONFIDENCE"]:
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
def run() -> dict:
    logger.info("=== NBA 预测 (24h窗口) ===")
    logger.info("当前时间: %s BJT", BJT.strftime("%Y-%m-%d %H:%M"))
    logger.info("预测窗口: %s ~ %s", BJT.strftime("%Y-%m-%d %H:%M"),
                WINDOW_END.strftime("%Y-%m-%d %H:%M"))

    # 1. 加载 ELO
    ratings = load_elo()
    logger.info("ELO 球队数: %d", len(ratings))

    # 2. 获取比赛数据
    all_games = fetch_odds_games()
    if not all_games:
        logger.warning("无比赛数据，返回空")
        return {"league": "nba", "status": "no_data", "predictions": [],
                "generated_at": BJT.isoformat(), "window": f"{BJT.strftime('%Y-%m-%d')}~{WINDOW_END.strftime('%Y-%m-%d %H:%M')}"}

    logger.info("API 返回总比赛数: %d", len(all_games))

    # 3. 24小时窗口过滤
    past_games, future_games = filter_24h_games(all_games)

    # 4. 回填已结束比赛结果，更新 ELO
    # 从 completed API 获取更准确的比分
    completed_api = fetch_completed_games()
    if completed_api:
        updated = update_elo_from_results(ratings, completed_api)
        if updated:
            logger.info("ELO 更新: %d 场完赛", updated)
            save_elo(ratings)

    # 5. 预测未来比赛
    predictions = []
    for g in future_games:
        try:
            home = g.get("home_team", "?")
            away = g.get("away_team", "?")
            if home == "?" or away == "?":
                continue
            odds_home, odds_away, spread_line, total_line = parse_odds(g, home)
            pred = predict_game(home, away, ratings, odds_home, odds_away, spread_line, total_line)
            pred["commence_time"] = g.get("commence_time", "")
            predictions.append(pred)
        except Exception as e:
            logger.warning("解析比赛异常，跳过: %s", e)
            continue

    # 6. 保存 ELO
    save_elo(ratings)

    # 6.5 AI 调整
    if _AI_ADJUSTMENTS and predictions:
        adjusted = 0
        for i, p in enumerate(predictions):
            before = p.get("win_prob", 0.5)
            p = adjust_prediction(p, _AI_ADJUSTMENTS)
            if p.get("ai_adjusted"):
                adjusted += 1
            predictions[i] = p
        if adjusted:
            logger.info("AI 调整: %d/%d 场已调整置信度", adjusted, len(predictions))

    # 7. 构建输出
    past_details = []
    for g in past_games:
        home = g.get("home_team", "?")
        away = g.get("away_team", "?")
        scores = g.get("scores", {})
        if isinstance(scores, dict):
            hs = scores.get(home, "?")
            as_ = scores.get(away, "?")
        else:
            hs = as_ = "?"
        score_str = f"{hs}-{as_}" if hs != "?" and as_ != "?" else ""
        past_details.append({
            "name": f"{to_cn(home)} vs {to_cn(away)}",
            "home": to_cn(home),
            "away": to_cn(away),
            "score": score_str,
        })

    output = {
        "league": "nba",
        "generated_at": BJT.isoformat(),
        "status": "ok",
        "window": f"{BJT.strftime('%Y-%m-%d %H:%M')}~{WINDOW_END.strftime('%Y-%m-%d %H:%M')}",
        "total_in_window": len(past_games) + len(future_games),
        "past_games_count": len(past_games),
        "future_games_count": len(future_games),
        "past_games_detail": past_details,
        "predictions": predictions,
        "predictions_count": len(predictions),
    }
    return output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    result = run()
    if not isinstance(result.get("predictions"), list):
        result["predictions"] = []
    print(json.dumps(result, ensure_ascii=False, indent=2))
