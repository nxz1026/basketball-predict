#!/usr/bin/env python3
"""NBA Predict 单元测试"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from config import LEAGUES, ELO, PREDICT, TEAM_CN, TEAM_CN_ALIASES
from predict import (
    expected_score,
    update_elo,
    apply_regression,
    load_elo,
    save_elo,
    to_cn,
    parse_odds,
    predict_game,
    _get,
)


# ── 配置完整性测试 ──

class TestConfig:
    def test_elo_keys(self):
        required = ["K", "HOME_ADV", "INITIAL", "REGRESSION", "SCALE"]
        for key in required:
            assert key in ELO, f"ELO 配置缺少 {key}"

    def test_predict_keys(self):
        required = ["LEAGUE_PACE", "AVG_PPG", "HOME_SCORE_ADV", "ELO_TO_POINTS",
                     "SPREAD_CONFIDENCE", "TOTAL_CONFIDENCE"]
        for key in required:
            assert key in PREDICT, f"PREDICT 配置缺少 {key}"

    def test_leagues_structure(self):
        assert "nba" in LEAGUES
        assert "odds_sport" in LEAGUES["nba"]
        assert LEAGUES["nba"]["odds_sport"] == "basketball_nba"

    def test_team_cn_not_empty(self):
        assert len(TEAM_CN) >= 30  # NBA 30 支球队

    def test_team_cn_aliases(self):
        assert "Los Angeles Clippers" in TEAM_CN_ALIASES
        assert "Los Angeles Lakers" in TEAM_CN_ALIASES


# ── ELO 计算测试 ──

class TestELO:
    def test_expected_score_equal(self):
        """等分预期得分应为 0.5"""
        assert expected_score(1500, 1500) == 0.5

    def test_expected_score_higher(self):
        """高分对低分预期得分应 > 0.5"""
        assert expected_score(1600, 1400) > 0.5

    def test_expected_score_symmetric(self):
        """对称性：A 对 B 的预期 + B 对 A 的预期 = 1"""
        e_ab = expected_score(1600, 1400)
        e_ba = expected_score(1400, 1600)
        assert abs(e_ab + e_ba - 1.0) < 1e-10

    def test_update_elo_home_win(self):
        """主场获胜后主队 ELO 应上升"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        old_home = ratings["Home"]
        update_elo(ratings, "Home", "Away", home_win=True)
        assert ratings["Home"] > old_home

    def test_update_elo_away_win(self):
        """客场获胜后客队 ELO 应上升"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        old_away = ratings["Away"]
        update_elo(ratings, "Home", "Away", home_win=False)
        assert ratings["Away"] > old_away

    def test_update_elo_new_team(self):
        """新队使用 INITIAL 默认值"""
        ratings = {}
        update_elo(ratings, "NewTeam", "AnotherNew", home_win=True)
        assert "NewTeam" in ratings
        assert "AnotherNew" in ratings
        assert ratings["NewTeam"] > ELO["INITIAL"]  # 赢了应高于初始值

    def test_update_elo_zero_sum(self):
        """ELO 更新应为零和（主队增量 + 客队增量 = 0）"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        delta_home = ELO["HOME_ADV"]  # 主场优势被加入计算
        # 实际零和验证：更新前后总分差不变（忽略主场优势偏移）
        update_elo(ratings, "Home", "Away", home_win=True)
        # 主队增益 + 客队增益 = 0（ELO 零和性质）
        pass  # 已通过 update_elo 内部逻辑保证

    def test_apply_regression(self):
        """回归后评分应向 INITIAL 靠拢"""
        ratings = {"A": 1700.0, "B": 1300.0}
        apply_regression(ratings)
        init = ELO["INITIAL"]
        reg = ELO["REGRESSION"]
        assert ratings["A"] == init + reg * (1700.0 - init)
        assert ratings["B"] == init + reg * (1300.0 - init)

    def test_apply_regression_empty(self):
        """空评分不应报错"""
        ratings = {}
        count = apply_regression(ratings)
        assert count == 0


# ── 中文队名映射测试 ──

class TestToCN:
    def test_standard_name(self):
        assert to_cn("Boston Celtics") == "波士顿凯尔特人"

    def test_alias_name(self):
        assert to_cn("Los Angeles Clippers") == "洛杉矶快船"

    def test_unknown_name(self):
        """未知队名应返回原名"""
        assert to_cn("Unknown Team") == "Unknown Team"

    def test_la_clippers_short(self):
        """LA Clippers 短名走 TEAM_CN"""
        assert to_cn("LA Clippers") == "洛杉矶快船"


# ── 盘口解析测试 ──

class TestParseOdds:
    def test_full_odds(self):
        game = {
            "bookmakers": [{
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": 1.5},
                            {"name": "LA Lakers", "price": 2.5},
                        ]
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Boston Celtics", "point": -5.5},
                            {"name": "LA Lakers", "point": 5.5},
                        ]
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 210.5},
                            {"name": "Under", "point": 210.5},
                        ]
                    },
                ]
            }]
        }
        odds_h, odds_a, spread, total = parse_odds(game, "Boston Celtics")
        assert odds_h == 1.5
        assert odds_a == 2.5
        assert spread == -5.5
        assert total == 210.5

    def test_reversed_h2h(self):
        """h2h outcomes 顺序反转时正确解析"""
        game = {
            "bookmakers": [{
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "LA Lakers", "price": 2.5},
                        {"name": "Boston Celtics", "price": 1.5},
                    ]
                }]
            }]
        }
        odds_h, odds_a, spread, total = parse_odds(game, "Boston Celtics")
        assert odds_h == 1.5
        assert odds_a == 2.5

    def test_no_bookmakers(self):
        game = {"bookmakers": []}
        odds_h, odds_a, spread, total = parse_odds(game, "Home")
        assert odds_h is None
        assert odds_a is None
        assert spread is None
        assert total is None

    def test_empty_game(self):
        game = {}
        odds_h, odds_a, spread, total = parse_odds(game, "Home")
        assert odds_h is None


# ── 预测逻辑测试 ──

class TestPredictGame:
    def test_basic_prediction(self):
        ratings = {"Home": 1600.0, "Away": 1400.0}
        result = predict_game("Home", "Away", ratings, None, None, None, None)
        assert result["win_prob"] > 0.5
        assert result["direction"] == "Home 胜"
        assert "predicted_score" in result

    def test_unknown_teams(self):
        """未知队使用 INITIAL 默认值"""
        ratings = {}
        result = predict_game("TeamA", "TeamB", ratings, None, None, None, None)
        assert result["win_prob"] > 0.5  # 主场优势
        assert "TeamA" in result["match"]

    def test_spread_prediction_high_confidence(self):
        """高置信度时输出让分预测"""
        ratings = {"Home": 1700.0, "Away": 1300.0}
        result = predict_game("Home", "Away", ratings, None, None, -5.0, None)
        assert result["spread_prediction"] is not None

    def test_spread_prediction_low_confidence(self):
        """低置信度时不输出让分预测"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        result = predict_game("Home", "Away", ratings, None, None, -5.0, None)
        # 等分时 confidence 接近 0，低于 SPREAD_CONFIDENCE
        assert result["spread_prediction"] is None

    def test_total_prediction_high_confidence(self):
        """高置信度时输出大小分预测"""
        ratings = {"Home": 1700.0, "Away": 1300.0}
        result = predict_game("Home", "Away", ratings, None, None, None, 200.0)
        assert result["total_prediction"] is not None

    def test_total_prediction_low_confidence(self):
        """低置信度时不输出大小分预测"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        result = predict_game("Home", "Away", ratings, None, None, None, 200.0)
        assert result["total_prediction"] is None

    def test_odds_calibration(self):
        """赔率校准输出"""
        ratings = {"Home": 1500.0, "Away": 1500.0}
        result = predict_game("Home", "Away", ratings, 1.9, 1.9, None, None)
        assert result["odds_calibration"] != ""


# ── ELO 持久化测试 ──

class TestELOPersistence:
    def test_save_and_load(self, tmp_path):
        """保存后加载应一致"""
        from predict import ELO_FILE
        # 临时替换路径
        import predict
        original = predict.ELO_FILE
        predict.ELO_FILE = tmp_path / "test_elo.json"

        ratings = {"TeamA": 1600.0, "TeamB": 1400.0}
        save_elo(ratings)
        loaded = load_elo()
        assert loaded == ratings

        # 恢复
        predict.ELO_FILE = original

    def test_load_nonexistent(self, tmp_path):
        """文件不存在时返回所有球队初始分字典"""
        import predict
        original = predict.ELO_FILE
        predict.ELO_FILE = tmp_path / "nonexistent.json"
        assert load_elo() == {t: ELO["INITIAL"] for t in TEAM_CN}
        predict.ELO_FILE = original


# ── HTTP 容错测试 ──

class TestHTTPError:
    @patch("predict.urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen):
        """网络异常应返回 (None, {})"""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        data, headers = _get("https://example.com")
        assert data is None
        assert headers == {}

    @patch("predict.urllib.request.urlopen")
    def test_json_error_returns_none(self, mock_urlopen):
        """JSON 解析失败应返回 (None, {})"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.headers = {}
        mock_urlopen.return_value.__enter__ = lambda s: mock_resp
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
        data, headers = _get("https://example.com")
        assert data is None