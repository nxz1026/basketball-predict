# NBA Predict 🏀

NBA 比赛预测，基于 ELO + 效率模型。

## 数据源

- **赛程 & 盘口**：[The Odds API](https://the-odds-api.com)（免费 500 次/月）
- **球队统计**：[balldontlie.io](https://balldontlie.io)（免费）

## 玩法匹配

| 玩法 | 模型 | 状态 |
|------|------|------|
| 胜负 | ELO win_prob | ✅ |
| 让分胜负 | ELO 分差预测 vs 让分线 | ✅ |
| 大小分 | 预测总分 vs 预设线 | ✅ |
| 胜分差 | 不适用，精度不足 | ❌ |

## 部署

- GHA 每天 21:00 BJT 自动运行
- 结果推送至飞书 webhook

## 配置

```bash
# 注册 https://the-odds-api.com 获取 key
# 设置 GitHub Secret: ODDS_API_KEY, FEISHU_WEBHOOK_URL
```
