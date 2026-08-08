"""NBA 预测配置"""

# The Odds API key — 注册 https://the-odds-api.com 获取
ODDS_API_KEY = ""

# 联赛
LEAGUES = {
    "nba": {
        "name": "NBA",
        "name_cn": "NBA",
        "odds_sport": "basketball_nba",
        "teams_file": "data/nba_teams.json",
    },
}

# ELO 配置
ELO = {
    "K": 32,              # K 因子
    "HOME_ADV": 3,        # 主场优势（ELO 分）
    "INITIAL": 1500,      # 初始分
    "REGRESSION": 0.5,    # 休赛期回归系数
    "SCALE": 400,         # ELO 标准分母（每 400 分差对应 10 倍胜率比）
}

# 预测配置
PREDICT = {
    "LEAGUE_PACE": 98.5,       # NBA 平均 pace（回合/48分钟）
    "AVG_PPG": 112.0,          # NBA 平均每场得分
    "HOME_SCORE_ADV": 2.5,     # 主场得分优势
    "ELO_TO_POINTS": 0.06,     # 每 1 ELO 分对应的得分差（每 10 ELO ≈ 0.6 分）
    "SPREAD_CONFIDENCE": 0.65, # 让分预测置信度阈值
    "TOTAL_CONFIDENCE": 0.60,  # 大小分置信度阈值
}

# 中文队名
TEAM_CN = {
    "Atlanta Hawks": "亚特兰大老鹰",
    "Boston Celtics": "波士顿凯尔特人",
    "Brooklyn Nets": "布鲁克林篮网",
    "Charlotte Hornets": "夏洛特黄蜂",
    "Chicago Bulls": "芝加哥公牛",
    "Cleveland Cavaliers": "克利夫兰骑士",
    "Dallas Mavericks": "达拉斯独行侠",
    "Denver Nuggets": "丹佛掘金",
    "Detroit Pistons": "底特律活塞",
    "Golden State Warriors": "金州勇士",
    "Houston Rockets": "休斯顿火箭",
    "Indiana Pacers": "印第安纳步行者",
    "LA Clippers": "洛杉矶快船",
    "Los Angeles Lakers": "洛杉矶湖人",
    "Memphis Grizzlies": "孟菲斯灰熊",
    "Miami Heat": "迈阿密热火",
    "Milwaukee Bucks": "密尔沃基雄鹿",
    "Minnesota Timberwolves": "明尼苏达森林狼",
    "New Orleans Pelicans": "新奥尔良鹈鹕",
    "New York Knicks": "纽约尼克斯",
    "Oklahoma City Thunder": "俄克拉荷马雷霆",
    "Orlando Magic": "奥兰多魔术",
    "Philadelphia 76ers": "费城76人",
    "Phoenix Suns": "菲尼克斯太阳",
    "Portland Trail Blazers": "波特兰开拓者",
    "Sacramento Kings": "萨克拉门托国王",
    "San Antonio Spurs": "圣安东尼奥马刺",
    "Toronto Raptors": "多伦多猛龙",
    "Utah Jazz": "犹他爵士",
    "Washington Wizards": "华盛顿奇才",
}

# 队名别名映射（API 可能返回不同写法）
TEAM_CN_ALIASES = {
    "Los Angeles Clippers": "洛杉矶快船",
    "Los Angeles Lakers": "洛杉矶湖人",
    "New Orleans Pelicans": "新奥尔良鹈鹕",
    "San Antonio Spurs": "圣安东尼奥马刺",
    "Golden State Warriors": "金州勇士",
    "Oklahoma City Thunder": "俄克拉荷马雷霆",
    "Portland Trail Blazers": "波特兰开拓者",
}