#!/usr/bin/env python3
"""NBA feedback_loop (reconcile / rolling / email summary) 单元测试"""

import json
from pathlib import Path

import ai.feedback_loop as fl


def _pred(match, home, away, direction, ai_adjusted=False):
    return {
        "match": match,
        "home": home,
        "away": away,
        "direction": direction,
        "win_prob": 0.7,
        "ai_adjusted": ai_adjusted,
        "ai_score_used": 80 if ai_adjusted else None,
    }


def _actual(name, home, away, score):
    return {"name": name, "home": home, "away": away, "score": score}


class TestReconcile:
    def test_correct_home_win(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        preds = [_pred("洛杉矶湖人 vs 波士顿凯尔特人", "Los Angeles Lakers",
                       "Boston Celtics", "洛杉矶湖人 胜", ai_adjusted=True)]
        actuals = [_actual("洛杉矶湖人 vs 波士顿凯尔特人", "洛杉矶湖人",
                            "波士顿凯尔特人", "112-108")]
        m = fl.reconcile_results(preds, actuals)
        assert m["total_matches"] == 1
        assert m["correct"] == 1
        assert m["accuracy"] == 1.0
        assert m["ai_adjusted_total"] == 1
        assert m["ai_adjusted_correct"] == 1

    def test_wrong_direction(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        preds = [_pred("洛杉矶湖人 vs 波士顿凯尔特人", "Los Angeles Lakers",
                       "Boston Celtics", "波士顿凯尔特人 胜")]
        actuals = [_actual("洛杉矶湖人 vs 波士顿凯尔特人", "洛杉矶湖人",
                            "波士顿凯尔特人", "112-108")]
        m = fl.reconcile_results(preds, actuals)
        assert m["correct"] == 0
        assert m["accuracy"] == 0.0

    def test_no_match_skipped(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        preds = [_pred("A vs B", "A", "B", "A 胜")]
        actuals = [_actual("C vs D", "C", "D", "99-88")]
        m = fl.reconcile_results(preds, actuals)
        assert m["total_matches"] == 0

    def test_uses_score_key_not_predicted_score(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        preds = [_pred("洛杉矶湖人 vs 波士顿凯尔特人", "Los Angeles Lakers",
                       "Boston Celtics", "洛杉矶湖人 胜")]
        # 旧字段名 predicted_score 也应兼容
        actuals = [{"name": "洛杉矶湖人 vs 波士顿凯尔特人",
                    "home": "洛杉矶湖人", "away": "波士顿凯尔特人",
                    "predicted_score": "112-108"}]
        m = fl.reconcile_results(preds, actuals)
        assert m["correct"] == 1


class TestRolling:
    def test_rolling_aggregates(self):
        history = [
            {"total_matches": 2, "correct": 1, "ai_adjusted_total": 1, "ai_adjusted_correct": 1},
            {"total_matches": 2, "correct": 2, "ai_adjusted_total": 0, "ai_adjusted_correct": 0},
        ]
        r = fl.compute_rolling_accuracy(history, window=30)
        assert r["total"] == 4
        assert r["correct"] == 3
        assert r["accuracy"] == 0.75

    def test_rolling_window(self):
        history = [{"total_matches": 10, "correct": 5}] * 5
        r = fl.compute_rolling_accuracy(history, window=2)
        assert r["total"] == 20
        assert r["correct"] == 10


class TestEmailSummary:
    def test_summary_includes_today(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        # 先写一条历史，便于测试滚动
        hist = [{"total_matches": 4, "correct": 3, "ai_adjusted_total": 0, "ai_adjusted_correct": 0}]
        fl.METRICS_FILE.write_text(json.dumps(hist), encoding="utf-8")
        preds = [_pred("洛杉矶湖人 vs 波士顿凯尔特人", "Los Angeles Lakers",
                       "Boston Celtics", "洛杉矶湖人 胜")]
        actuals = [_actual("洛杉矶湖人 vs 波士顿凯尔特人", "洛杉矶湖人",
                            "波士顿凯尔特人", "112-108")]
        m = fl.reconcile_results(preds, actuals)
        s = fl.format_email_summary(m)
        assert "命中率小结" in s
        assert "本日命中率" in s
        assert "近30次累计命中率" in s

    def test_summary_no_backfill(self, tmp_path):
        fl.METRICS_FILE = tmp_path / "metrics.json"
        m = {"total_matches": 0, "correct": 0, "accuracy": 0}
        s = fl.format_email_summary(m)
        assert "本日无回填比赛可供核对" in s
