from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import feedparser
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import yagmail
except Exception:
    yagmail = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None
    A4 = None

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None

try:
    import bcrypt
except Exception:
    bcrypt = None


APP_TITLE = "📊 Analyst Dashboard v6.2 - Independent App"
DEFAULT_MODEL = "gpt-4.1-mini"

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"
CACHE_FILE = DATA_DIR / "ai_cache.json"
EMAIL_LOG_FILE = DATA_DIR / "email_send_log.csv"
MARKET_HISTORY_FILE = DATA_DIR / "market_history.csv"
IC_HISTORY_FILE = DATA_DIR / "ic_note_history.csv"
TRADE_IDEAS_HISTORY_FILE = DATA_DIR / "trade_ideas_history.csv"
USERS_FILE = DATA_DIR / "users.json"
RECIPIENT_GROUPS_FILE = DATA_DIR / "recipient_groups.json"
AUDIT_LOG_FILE = DATA_DIR / "audit_log.csv"

DEFAULT_TICKERS = {
    "VNINDEX": "^VNINDEX",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
    "USD Index": "DX-Y.NYB",
    "USD/VND (proxy)": "VND=X",
}

DEFAULT_VN_PRODUCTS = {
    "VN Equity - Large Cap": ["FPT.VN", "VCB.VN", "HPG.VN", "MBB.VN", "SSI.VN", "VHM.VN", "GAS.VN"],
    "VN Broker Favorites": ["FPT.VN", "MWG.VN", "ACB.VN", "TCB.VN", "VCB.VN", "SSI.VN", "HPG.VN", "PNJ.VN"],
}

RSS_FEEDS = {
    "VnExpress Kinh doanh": "https://vnexpress.net/rss/kinh-doanh.rss",
    "VnExpress Chứng khoán": "https://vnexpress.net/rss/chung-khoan.rss",
    "Vietstock Chứng khoán": "https://vietstock.vn/rss/chung-khoan.rss",
    "Vietstock Tài chính": "https://vietstock.vn/rss/tai-chinh.rss",
    "Vietstock Doanh nghiệp": "https://vietstock.vn/rss/doanh-nghiep.rss",
    "Reuters Markets": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters World": "https://feeds.reuters.com/Reuters/worldNews",
    "MarketWatch Top Stories": "https://feeds.marketwatch.com/marketwatch/topstories/",
}

ROLE_PERMISSIONS = {
    "admin": {
        "view_dashboard", "run_pipeline", "generate_notes", "send_email", "export_notes",
        "manage_users", "manage_groups", "view_logs", "manage_schedule"
    },
    "analyst": {
        "view_dashboard", "run_pipeline", "generate_notes", "send_email", "export_notes"
    },
    "viewer": {
        "view_dashboard"
    },
}

SCHEDULER_STARTED = False


# =========================================================
# HELPERS
# =========================================================
def init_csv(path: Path, columns: List[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def get_runtime_value(config_value: str, secret_name: str) -> str:
    secret_val = get_secret(secret_name, "")
    return secret_val if secret_val else (config_value or "")


def ai_is_available(api_key: str) -> bool:
    return bool(api_key) and (OpenAI is not None)


def load_cache() -> dict:
    return load_json(CACHE_FILE, {})


def save_cache(cache: dict) -> None:
    save_json(CACHE_FILE, cache)


def append_csv_row(path: Path, row: dict) -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_files() -> None:
    if not CONFIG_FILE.exists():
        save_json(CONFIG_FILE, {
            "model": DEFAULT_MODEL,
            "house_view": "Neutral",
            "default_recipients": "",
            "smart_mode": True,
            "auto_send_after_run_all": False,
            "trade_ideas_count": 7,
            "auto_send_enabled": False,
            "auto_send_time": "09:00",
            "auto_send_groups": ["internal_morning"],
            "auto_user_news": "",
            "auto_expert_notes": ""
        })

    if not CACHE_FILE.exists():
        save_json(CACHE_FILE, {})

    if not USERS_FILE.exists():
        default_password_hash = hash_password("admin123")
        save_json(USERS_FILE, [
            {
                "username": "admin",
                "password_hash": default_password_hash,
                "role": "admin",
                "full_name": "System Admin",
                "active": True
            }
        ])

    if not RECIPIENT_GROUPS_FILE.exists():
        save_json(RECIPIENT_GROUPS_FILE, {
            "internal_morning": [],
            "cio_team": [],
            "pm_team": []
        })

    init_csv(EMAIL_LOG_FILE, ["SendTime", "Sender", "Recipients", "Subject", "Status", "Error"])
    init_csv(MARKET_HISTORY_FILE, ["Date", "MorningNote", "ClosingNote"])
    init_csv(IC_HISTORY_FILE, ["Date", "ICNote"])
    init_csv(TRADE_IDEAS_HISTORY_FILE, ["Date", "TradeIdeas"])
    init_csv(AUDIT_LOG_FILE, ["Time", "Username", "Action", "Detail"])


def load_config() -> dict:
    return load_json(CONFIG_FILE, {})


def save_config(cfg: dict) -> None:
    save_json(CONFIG_FILE, cfg)


def log_audit(username: str, action: str, detail: str = "") -> None:
    append_csv_row(AUDIT_LOG_FILE, {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username": username,
        "Action": action,
        "Detail": detail,
    })


# =========================================================
# AUTH
# =========================================================
def hash_password(password: str) -> str:
    if bcrypt is not None:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("$2") and bcrypt is not None:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


def load_users() -> List[dict]:
    return load_json(USERS_FILE, [])


def save_users(users: List[dict]) -> None:
    save_json(USERS_FILE, users)


def authenticate(username: str, password: str) -> dict | None:
    for user in load_users():
        if user.get("username") == username and user.get("active", True):
            if verify_password(password, user.get("password_hash", "")):
                return user
    return None


def has_permission(permission: str) -> bool:
    user = st.session_state.get("current_user")
    if not user:
        return False
    role = user.get("role", "viewer")
    return permission in ROLE_PERMISSIONS.get(role, set())


# =========================================================
# GROUPS
# =========================================================
def load_recipient_groups() -> dict:
    return load_json(RECIPIENT_GROUPS_FILE, {})


def save_recipient_groups(groups: dict) -> None:
    save_json(RECIPIENT_GROUPS_FILE, groups)


def get_emails_from_groups(group_names: List[str]) -> List[str]:
    groups = load_recipient_groups()
    emails: List[str] = []
    for g in group_names:
        emails.extend(groups.get(g, []))
    return sorted(list(set([e.strip() for e in emails if str(e).strip()])))


# =========================================================
# AI
# =========================================================
def generate_with_openai(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1800) -> str:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return getattr(response, "output_text", "").strip()


def cached_ai_call(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1800) -> str:
    if not ai_is_available(api_key):
        return ""
    cache = load_cache()
    key = hash_text(model + system_prompt + user_prompt + str(max_output_tokens))
    if key in cache:
        return cache[key]
    try:
        out = generate_with_openai(api_key, model, system_prompt, user_prompt, max_output_tokens)
        if out:
            cache[key] = out
            save_cache(cache)
        return out
    except Exception:
        return ""


# =========================================================
# MARKET / NEWS
# =========================================================
def fetch_market_snapshot(tickers: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="1d", auto_adjust=False)
            if hist.empty or len(hist) < 2:
                continue
            last_close = float(hist.iloc[-1]["Close"])
            prev_close = float(hist.iloc[-2]["Close"])
            change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else None
            rows.append({
                "Asset": name,
                "Ticker": ticker,
                "Price": round(last_close, 2),
                "ChangePct": round(change_pct, 2) if change_pct is not None else None,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_vn_recommendation_watchlist() -> pd.DataFrame:
    tickers = DEFAULT_VN_PRODUCTS.get("VN Broker Favorites", [])
    rows = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
            if hist.empty or len(hist) < 2:
                continue
            last_close = float(hist.iloc[-1]["Close"])
            prev_close = float(hist.iloc[-2]["Close"])
            change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else None
            rows.append({
                "Ticker": ticker,
                "Price": round(last_close, 2),
                "ChangePct": round(change_pct, 2) if change_pct is not None else None,
                "BrokerView": "Theo dõi / Watch",
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="ChangePct", ascending=False).reset_index(drop=True)
    return df


def detect_region(source: str, title: str, summary: str) -> str:
    source_l = (source or "").lower()
    text = f"{source_l} {(title or '').lower()} {(summary or '').lower()}"
    vn_source_keywords = ["vnexpress", "vietstock", "cafef", "ndh", "stockbiz"]
    vn_content_keywords = [
        "vietnam", "viet nam", "vnindex", "hose", "hnx", "upcom", "sbv",
        "tỷ giá", "ty gia", "lãi suất", "trái phiếu", "chứng khoán",
        "ngân hàng", "vnd", "fpt", "vcb", "hpg", "ssi"
    ]
    if any(k in source_l for k in vn_source_keywords):
        return "Vietnam"
    if any(k in text for k in vn_content_keywords):
        return "Vietnam"
    return "Global"


def classify_asset_class(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["bond", "yield", "fixed income", "trái phiếu", "lãi suất", "us 10y"]):
        return "Fixed Income"
    if any(k in t for k in ["oil", "gold", "commodity", "dầu", "vàng"]):
        return "Commodity"
    if any(k in t for k in ["usd", "dxy", "fx", "forex", "tỷ giá", "ty gia"]):
        return "Commodity / FX"
    return "Equity"


def estimate_vn_impact(text: str) -> int:
    t = (text or "").lower()
    score = 1
    for k in ["vnindex", "vietnam", "sbv", "usd", "oil", "bank", "fpt", "vcb", "hpg"]:
        if k in t:
            score += 1
    return min(score, 5)


def fetch_rss_news(feed_map: Dict[str, str], max_per_feed: int = 6) -> pd.DataFrame:
    rows = []
    for source_name, url in feed_map.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                summary = re.sub("<.*?>", "", entry.get("summary", "")).strip() if entry.get("summary") else ""
                link = entry.get("link", "").strip()
                full_text = f"{title} {summary}"
                rows.append({
                    "Region": detect_region(source_name, title, summary),
                    "Source": source_name,
                    "Title": title,
                    "Summary": summary,
                    "Link": link,
                    "AssetClass": classify_asset_class(full_text),
                    "VNImpact": estimate_vn_impact(full_text),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def filter_news_for_ai(news_df: pd.DataFrame, smart_mode: bool = True) -> pd.DataFrame:
    if news_df.empty:
        return news_df
    if not smart_mode:
        return news_df.copy()
    return news_df[news_df["VNImpact"] >= 2].copy()


def split_news_by_region(news_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if news_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    return news_df[news_df["Region"] == "Vietnam"].copy(), news_df[news_df["Region"] == "Global"].copy()


def build_market_highlights(market_df: pd.DataFrame) -> str:
    if market_df.empty:
        return "Không có dữ liệu thị trường.\nNo market data available."
    return "\n".join(f"- {r['Asset']}: {r['Price']} ({r['ChangePct']}%)" for _, r in market_df.iterrows())


def build_vn_global_news_brief(news_df: pd.DataFrame, vn_top_n: int = 4, global_top_n: int = 4) -> str:
    vn_df, global_df = split_news_by_region(news_df)
    lines = ["TIN VIỆT NAM / VIETNAM NEWS:"]
    if vn_df.empty:
        lines.append("- Không có. / None.")
    else:
        for _, r in vn_df.sort_values(by="VNImpact", ascending=False).head(vn_top_n).iterrows():
            lines.append(f"- [{r['AssetClass']}] {r['Title']}")
    lines.append("")
    lines.append("TIN QUỐC TẾ / GLOBAL NEWS:")
    if global_df.empty:
        lines.append("- Không có. / None.")
    else:
        for _, r in global_df.sort_values(by="VNImpact", ascending=False).head(global_top_n).iterrows():
            lines.append(f"- [{r['AssetClass']}] {r['Title']}")
    return "\n".join(lines)


def build_top_actionable_signals(news_df: pd.DataFrame, top_n: int = 5) -> str:
    if news_df.empty:
        return "- Không có tín hiệu. / No actionable signal."
    df = news_df.sort_values(by="VNImpact", ascending=False).head(top_n)
    return "\n".join(f"- [{r['AssetClass']}] {r['Title']} | VN Impact {r['VNImpact']}" for _, r in df.iterrows())


def get_market_bias(market_df: pd.DataFrame) -> str:
    if market_df.empty:
        return "Neutral"
    vals = []
    for asset in ["VNINDEX", "Dow Jones", "Nasdaq", "S&P 500"]:
        row = market_df.loc[market_df["Asset"] == asset]
        if not row.empty and pd.notna(row.iloc[0]["ChangePct"]):
            vals.append(float(row.iloc[0]["ChangePct"]))
    if not vals:
        return "Neutral"
    avg = sum(vals) / len(vals)
    if avg >= 1:
        return "Bullish"
    if avg >= 0.2:
        return "Slightly Bullish"
    if avg <= -1:
        return "Bearish"
    if avg <= -0.2:
        return "Slightly Bearish"
    return "Neutral"


# =========================================================
# EXPERT / ALLOCATION / IDEAS
# =========================================================
def build_expert_fund_summary(user_expert_notes: str = "") -> str:
    base = """
- SSI: Ưu tiên large caps và câu chuyện nâng hạng / Focus on large caps and the upgrade story.
- VNDirect: Theo dõi lãi suất, tỷ giá và nhóm dẫn dắt / Monitor rates, FX and leadership groups.
- Funds: Ưu tiên allocation linh hoạt / Prefer flexible allocation.
"""
    if normalize_text(user_expert_notes):
        base += "\n- Additional notes:\n" + normalize_text(user_expert_notes)[:1200]
    return base


def _get_change(market_df: pd.DataFrame, asset_name: str) -> float:
    row = market_df.loc[market_df["Asset"] == asset_name]
    if row.empty or pd.isna(row.iloc[0]["ChangePct"]):
        return 0.0
    return float(row.iloc[0]["ChangePct"])


def compute_portfolio_allocation(market_df: pd.DataFrame, news_df: pd.DataFrame) -> dict:
    allocation = {
        "Equity": {"Global": "Neutral", "Vietnam": "Neutral"},
        "FixedIncome": {"Global": "Neutral", "Vietnam": "Neutral"},
        "Commodity": {"Oil": "Neutral", "Gold": "Neutral"},
        "FX": {"USD": "Neutral"},
        "Regime": "Neutral / Transition"
    }
    if market_df.empty:
        return allocation

    oil_chg = _get_change(market_df, "Oil (WTI)")
    gold_chg = _get_change(market_df, "Gold")
    usd_chg = _get_change(market_df, "USD Index")
    vn_chg = _get_change(market_df, "VNINDEX")
    spx_chg = _get_change(market_df, "S&P 500")
    us10y_chg = _get_change(market_df, "US 10Y Yield")

    score = 0
    if oil_chg > 1:
        allocation["Commodity"]["Oil"] = "Overweight"
        allocation["FixedIncome"]["Global"] = "Underweight"
        score -= 1
    if gold_chg > 0.5:
        allocation["Commodity"]["Gold"] = "Overweight"
    if usd_chg > 0.5:
        allocation["FX"]["USD"] = "Long"
        allocation["Equity"]["Global"] = "Underweight"
        score -= 1
    if us10y_chg > 0.5:
        allocation["FixedIncome"]["Global"] = "Underweight"
    elif us10y_chg < -0.3:
        allocation["FixedIncome"]["Global"] = "Overweight"
    if vn_chg > 1:
        allocation["Equity"]["Vietnam"] = "Overweight"
        score += 1
    elif vn_chg < -1:
        allocation["Equity"]["Vietnam"] = "Underweight"
    if spx_chg > 0.5:
        allocation["Equity"]["Global"] = "Overweight"
        score += 1
    elif spx_chg < -0.8:
        allocation["Equity"]["Global"] = "Underweight"

    if score >= 1 and usd_chg <= 0.5:
        allocation["Regime"] = "Risk-on"
    elif score <= -1 or usd_chg > 0.5 or oil_chg > 1.5:
        allocation["Regime"] = "Risk-off"

    return allocation


def format_allocation(allocation: dict) -> str:
    lines = [f"PORTFOLIO ALLOCATION / PHÂN BỔ DANH MỤC",
             f"- Market Regime / Chế độ thị trường: {allocation.get('Regime', 'Neutral')}"]
    lines.append("")
    lines.append("EQUITY:")
    for k, v in allocation.get("Equity", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("FIXED INCOME:")
    for k, v in allocation.get("FixedIncome", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("COMMODITY:")
    for k, v in allocation.get("Commodity", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("FX:")
    for k, v in allocation.get("FX", {}).items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def score_trade_idea(asset: str, market_df: pd.DataFrame, news_df: pd.DataFrame) -> dict:
    score = 5.0
    rationale = []
    oil_chg = _get_change(market_df, "Oil (WTI)")
    gold_chg = _get_change(market_df, "Gold")
    usd_chg = _get_change(market_df, "USD Index")
    vn_chg = _get_change(market_df, "VNINDEX")
    spx_chg = _get_change(market_df, "S&P 500")
    a = asset.lower()

    if "oil" in a and oil_chg > 1:
        score += 2
        rationale.append("Oil momentum positive")
    if "gold" in a and (gold_chg > 0.5 or usd_chg > 0.4):
        score += 1
        rationale.append("Defensive demand")
    if "usd" in a and usd_chg > 0.5:
        score += 2
        rationale.append("USD strength confirmed")
    if ("vnindex" in a or ".vn" in a or "vietnam" in a) and vn_chg > 0.7:
        score += 1.5
        rationale.append("Vietnam momentum supportive")
    if "global equity" in a and spx_chg > 0.5:
        score += 1.5

    if not news_df.empty:
        matches = news_df[
            news_df["Title"].str.contains(asset.split()[0], case=False, na=False) |
            news_df["Summary"].str.contains(asset.split()[0], case=False, na=False)
        ]
        score += min(len(matches) * 0.5, 2.0)

    score = max(1.0, min(score, 10.0))
    confidence = "High" if score >= 8 else "Medium" if score >= 6 else "Low"
    direction = "Overweight / Tăng tỷ trọng" if score >= 8 else "Positive Watch / Theo dõi tích cực" if score >= 6.5 else "Watch"
    horizon = "1-4 weeks" if score >= 7 else "Short-term watch"
    return {
        "Asset": asset,
        "Score": round(score, 1),
        "Confidence": confidence,
        "Direction": direction,
        "Horizon": horizon,
        "Rationale": "; ".join(rationale) if rationale else "Neutral setup"
    }


def generate_ranked_trade_ideas(market_df: pd.DataFrame, news_df: pd.DataFrame, ideas_count: int = 7) -> pd.DataFrame:
    universe = ["Oil", "Gold", "USD", "VNINDEX", "Vietnam Banks", "FPT.VN", "HPG.VN", "VCB.VN", "SSI.VN", "Global Equity"]
    scored = [score_trade_idea(asset, market_df, news_df) for asset in universe]
    df = pd.DataFrame(scored).sort_values(by="Score", ascending=False).head(ideas_count).reset_index(drop=True)
    df.index = df.index + 1
    return df


def format_trade_ideas_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "Chưa có trade idea / No trade ideas."
    lines = ["TRADE IDEAS (RANKED) / Ý TƯỞNG ĐẦU TƯ"]
    for idx, row in df.iterrows():
        lines.append(
            f"\n#{idx} {row['Asset']}\n"
            f"- Direction: {row['Direction']}\n"
            f"- Score: {row['Score']}\n"
            f"- Confidence: {row['Confidence']}\n"
            f"- Horizon: {row['Horizon']}\n"
            f"- Rationale: {row['Rationale']}"
        )
    return "\n".join(lines)


def extract_top_idea(trade_ideas_text: str) -> str:
    text = normalize_text(trade_ideas_text)
    if not text:
        return "Chưa có ý tưởng nổi bật / No top idea available."
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        return blocks[1]
    return "\n".join(text.splitlines()[:8])


# =========================================================
# NOTES
# =========================================================
def fallback_morning_note(note_date: str, bias: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""MORNING NOTE / BẢN TIN SÁNG - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Theo dõi lợi suất và lạm phát.
- Monitor yields and inflation.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá và thanh khoản.
- Monitor FX and liquidity.

3. Global Equity / Equity toàn cầu
- Tâm lý phụ thuộc risk-on/risk-off.
- Sentiment depends on risk-on/risk-off.

4. Vietnam Equity / Equity Việt Nam
- Tập trung nhóm dẫn dắt và dòng tiền.
- Focus on leadership groups and flows.

5. Commodity / FX
- Dầu, vàng, USD là biến số lớn.
- Oil, gold and USD are major variables.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio Positioning & CIO View
- House view: {bias}
- Ưu tiên allocation linh hoạt.
- Keep allocation flexible.

{allocation_text}

8. Key News
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def fallback_closing_note(note_date: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""CLOSING NOTE / BẢN TIN CUỐI NGÀY - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Đánh giá lại lợi suất.
- Reassess yields.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá và thanh khoản.
- Monitor FX and liquidity.

3. Global Equity / Equity toàn cầu
- Kiểm tra risk sentiment.
- Check risk sentiment.

4. Vietnam Equity / Equity Việt Nam
- Kiểm tra độ rộng và thanh khoản.
- Check breadth and liquidity.

5. Commodity / FX
- Theo dõi dầu, vàng, USD.
- Monitor oil, gold, USD.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Changes vs Morning / Next Plan
{allocation_text}
"""


def fallback_ic_note(note_date: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""IC NOTE / GHI CHÚ IC - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Duration trung lập.
- Neutral duration.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi đường cong lợi suất.
- Monitor local yield curve.

3. Global Equity / Equity toàn cầu
- Positioning linh hoạt.
- Flexible positioning.

4. Vietnam Equity / Equity Việt Nam
- Ưu tiên large caps nếu dòng tiền hỗ trợ.
- Prefer large caps if flows support.

5. Commodity / FX
- Theo dõi dầu, vàng, USD.
- Monitor oil, gold, USD.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio implication / Hàm ý danh mục
{allocation_text}

8. Key News
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def generate_morning_note(api_key: str, model: str, note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are a senior CIO strategist.
Write a STRICTLY BILINGUAL Vietnamese-English Morning Note.
Focus on interpretation, portfolio action, and cross-asset implications.
"""
    user_prompt = f"""
Date: {note_date}
House View: {bias}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

SIGNALS:
{build_top_actionable_signals(news_df, 5)}

EXPERT NOTES:
{build_expert_fund_summary(expert_notes)}

ALLOCATION:
{allocation_text}

USER NOTES:
{normalize_text(user_news)[:700]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1800)
    return out if out else fallback_morning_note(note_date, bias, news_df, expert_notes, allocation_text)


def generate_closing_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are an end-of-day strategist.
Write a STRICTLY BILINGUAL Vietnamese-English Closing Note.
"""
    user_prompt = f"""
Date: {note_date}
MARKET:
{build_market_highlights(market_df)}
NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}
EXPERT NOTES:
{build_expert_fund_summary(expert_notes)}
ALLOCATION:
{allocation_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1800)
    return out if out else fallback_closing_note(note_date, news_df, expert_notes, allocation_text)


def generate_ic_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are a PM strategist writing for CIO and Investment Committee.
Write a STRICTLY BILINGUAL Vietnamese-English IC Note.
"""
    user_prompt = f"""
Date: {note_date}
MARKET:
{build_market_highlights(market_df)}
NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}
EXPERT NOTES:
{build_expert_fund_summary(expert_notes)}
ALLOCATION:
{allocation_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1900)
    return out if out else fallback_ic_note(note_date, news_df, expert_notes, allocation_text)


# =========================================================
# EXPORT
# =========================================================
def export_note_to_docx(title: str, content: str) -> bytes | None:
    if Document is None:
        return None
    doc = Document()
    doc.add_heading(title, level=1)
    for line in content.split("\n"):
        doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()


def export_note_to_pdf(title: str, content: str) -> bytes | None:
    if canvas is None or A4 is None:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title[:90])
    y -= 24
    c.setFont("Helvetica", 10)
    for raw_line in content.split("\n"):
        line = raw_line[:120]
        if y < 40:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)
        c.drawString(40, y, line)
        y -= 14
    c.save()
    buffer.seek(0)
    return buffer.read()


# =========================================================
# EMAIL
# =========================================================
def generate_email_body(api_key: str, model: str, core_note: str, report_date: str, run_label: str) -> Tuple[str, str]:
    subject = f"[{run_label}] Cập nhật thị trường | Market Update - {report_date}"
    system_prompt = """Write a STRICTLY BILINGUAL Vietnamese-English market update email. Keep it concise and institutional."""
    out = cached_ai_call(api_key, model, system_prompt, core_note[:3500], 1200)
    if out:
        return subject, out
    return subject, f"Kính gửi anh/chị,\nDear all,\n\n{core_note}\n\nTrân trọng.\nBest regards."


def send_email_yagmail(sender: str, password: str, recipients: List[str], subject: str, body: str):
    if yagmail is None:
        return False, "Chưa cài yagmail."
    if not sender or not password or not recipients:
        return False, "Thiếu email gửi / mật khẩu / người nhận."
    try:
        yag = yagmail.SMTP(user=sender, password=password)
        yag.send(to=recipients, subject=subject, contents=body)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def log_email_send(send_time: str, sender: str, recipients: List[str], subject: str, status: str, error_message: str = "") -> None:
    append_csv_row(EMAIL_LOG_FILE, {
        "SendTime": send_time,
        "Sender": sender,
        "Recipients": ", ".join(recipients),
        "Subject": subject,
        "Status": status,
        "Error": error_message,
    })


def save_market_history(note_date: str, morning_note: str = "", closing_note: str = "") -> None:
    append_csv_row(MARKET_HISTORY_FILE, {"Date": note_date, "MorningNote": morning_note[:6000], "ClosingNote": closing_note[:6000]})


def save_ic_history(note_date: str, ic_note: str) -> None:
    append_csv_row(IC_HISTORY_FILE, {"Date": note_date, "ICNote": ic_note[:6000]})


def save_trade_ideas_history(note_date: str, trade_ideas: str) -> None:
    append_csv_row(TRADE_IDEAS_HISTORY_FILE, {"Date": note_date, "TradeIdeas": trade_ideas[:6000]})


def build_email_bundle(api_key: str, model: str, report_date: str, run_label: str, core_note: str, ic_note: str, trade_ideas: str, allocation_text: str) -> Tuple[str, str]:
    subject, body = generate_email_body(api_key, model, core_note + "\n\n" + allocation_text, report_date, run_label)
    top_idea = extract_top_idea(trade_ideas)
    if allocation_text:
        body += "\n\n=== Portfolio Allocation ===\n" + allocation_text
    if top_idea:
        body += "\n\n=== Top Trade Idea ===\n" + top_idea
    if trade_ideas:
        body += "\n\n=== Full Trade Ideas ===\n" + trade_ideas
    if ic_note:
        body += "\n\n=== IC Note ===\n" + ic_note
    return subject, body


# =========================================================
# PIPELINE
# =========================================================
def run_all_pipeline(api_key: str, model: str, report_date: str, cfg: dict, user_news: str, expert_notes: str):
    market_df = fetch_market_snapshot(DEFAULT_TICKERS)
    news_df = fetch_rss_news(RSS_FEEDS, max_per_feed=6)
    news_df_ai = filter_news_for_ai(news_df, smart_mode=cfg.get("smart_mode", True))

    allocation = compute_portfolio_allocation(market_df, news_df_ai)
    allocation_text = format_allocation(allocation)

    trade_ideas_df = generate_ranked_trade_ideas(market_df, news_df_ai, int(cfg.get("trade_ideas_count", 7)))
    trade_ideas_text = format_trade_ideas_df(trade_ideas_df)

    morning_note = generate_morning_note(api_key, model, report_date, cfg.get("house_view", "Neutral"), market_df, news_df_ai, user_news, expert_notes, allocation_text)
    closing_note = generate_closing_note(api_key, model, report_date, market_df, news_df_ai, expert_notes, allocation_text)
    ic_note = generate_ic_note(api_key, model, report_date, market_df, news_df_ai, expert_notes, allocation_text)

    email_subject, email_body = build_email_bundle(api_key, model, report_date, "MANUAL", morning_note, ic_note, trade_ideas_text, allocation_text)

    return {
        "market_df": market_df,
        "news_df": news_df,
        "news_df_ai": news_df_ai,
        "allocation_text": allocation_text,
        "trade_ideas_df": trade_ideas_df,
        "morning_note": morning_note,
        "closing_note": closing_note,
        "ic_note": ic_note,
        "trade_ideas": trade_ideas_text,
        "email_subject": email_subject,
        "email_body": email_body,
    }


# =========================================================
# AUTO SCHEDULE 09:00
# =========================================================
def scheduled_morning_job():
    cfg = load_config()
    api_key = get_secret("OPENAI_API_KEY", "")
    sender = get_secret("SENDER_EMAIL", "")
    sender_pw = get_secret("SENDER_PASSWORD", "")
    model = cfg.get("model", DEFAULT_MODEL)

    groups = cfg.get("auto_send_groups", ["internal_morning"])
    recipients = get_emails_from_groups(groups)
    if not recipients:
        return

    report_date = datetime.now().strftime("%Y-%m-%d")
    result = run_all_pipeline(api_key, model, report_date, cfg, cfg.get("auto_user_news", ""), cfg.get("auto_expert_notes", ""))
    subject, body = build_email_bundle(api_key, model, report_date, "09:00 AUTO", result["morning_note"], result["ic_note"], result["trade_ideas"], result["allocation_text"])

    ok, msg = send_email_yagmail(sender, sender_pw, recipients, subject, body)
    log_email_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sender, recipients, subject, "OK" if ok else "ERROR", "" if ok else msg)
    save_market_history(report_date, result["morning_note"], result["closing_note"])
    save_ic_history(report_date, result["ic_note"])
    save_trade_ideas_history(report_date, result["trade_ideas"])


def start_scheduler_if_needed():
    global SCHEDULER_STARTED
    if SCHEDULER_STARTED or BackgroundScheduler is None:
        return
    cfg = load_config()
    if not cfg.get("auto_send_enabled", False):
        return
    run_time = cfg.get("auto_send_time", "09:00")
    try:
        hour, minute = [int(x) for x in run_time.split(":")]
    except Exception:
        hour, minute = 9, 0
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_morning_job, "cron", hour=hour, minute=minute, id="daily_morning_email", replace_existing=True)
    scheduler.start()
    SCHEDULER_STARTED = True


# =========================================================
# UI
# =========================================================
def init_session():
    defaults = {
        "current_user": None,
        "market_df": pd.DataFrame(),
        "news_df": pd.DataFrame(),
        "news_df_ai": pd.DataFrame(),
        "trade_ideas_df": pd.DataFrame(),
        "allocation_text": "",
        "morning_note": "",
        "closing_note": "",
        "ic_note": "",
        "trade_ideas": "",
        "email_subject": "",
        "email_body": "",
        "user_news": "",
        "expert_notes": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def login_screen():
    st.title(APP_TITLE)
    st.subheader("Đăng nhập / Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        user = authenticate(username, password)
        if user:
            st.session_state["current_user"] = user
            log_audit(user.get("username", ""), "login")
            st.rerun()
        else:
            st.error("Sai tài khoản hoặc mật khẩu.")


def render_export_buttons(title_prefix: str, content: str, filename_prefix: str):
    if not content:
        return
    c1, c2 = st.columns(2)
    with c1:
        docx_bytes = export_note_to_docx(title_prefix, content)
        if docx_bytes:
            st.download_button(
                "Export Word",
                data=docx_bytes,
                file_name=f"{filename_prefix}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
    with c2:
        pdf_bytes = export_note_to_pdf(title_prefix, content)
        if pdf_bytes:
            st.download_button(
                "Export PDF",
                data=pdf_bytes,
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


def admin_users_tab():
    st.markdown("### User Management")
    users = load_users()
    st.dataframe(pd.DataFrame(users), use_container_width=True)

    st.markdown("#### Tạo user mới")
    new_username = st.text_input("Username mới", key="new_username")
    new_full_name = st.text_input("Full name", key="new_full_name")
    new_password = st.text_input("Password mới", type="password", key="new_password")
    new_role = st.selectbox("Role", ["admin", "analyst", "viewer"], key="new_role")
    new_active = st.checkbox("Active", value=True, key="new_active")

    if st.button("Create User", use_container_width=True):
        if new_username and new_password:
            if any(u.get("username") == new_username for u in users):
                st.error("Username đã tồn tại.")
            else:
                users.append({
                    "username": new_username,
                    "password_hash": hash_password(new_password),
                    "role": new_role,
                    "full_name": new_full_name,
                    "active": new_active
                })
                save_users(users)
                log_audit(st.session_state["current_user"]["username"], "create_user", new_username)
                st.success("Đã tạo user.")
                st.rerun()

    st.markdown("#### Cập nhật user")
    if users:
        selected_username = st.selectbox("Chọn user", [u["username"] for u in users], key="edit_user_select")
        selected = next((u for u in users if u["username"] == selected_username), None)
        if selected:
            edit_role = st.selectbox("Role mới", ["admin", "analyst", "viewer"], index=["admin", "analyst", "viewer"].index(selected.get("role", "viewer")), key="edit_role")
            edit_active = st.checkbox("Active", value=selected.get("active", True), key="edit_active")
            reset_password = st.text_input("Reset password (nếu cần)", type="password", key="reset_password")

            if st.button("Update User", use_container_width=True):
                for u in users:
                    if u["username"] == selected_username:
                        u["role"] = edit_role
                        u["active"] = edit_active
                        if reset_password:
                            u["password_hash"] = hash_password(reset_password)
                save_users(users)
                log_audit(st.session_state["current_user"]["username"], "update_user", selected_username)
                st.success("Đã cập nhật user.")
                st.rerun()


def admin_groups_tab():
    st.markdown("### Recipient Groups")
    groups = load_recipient_groups()
    group_rows = [{"Group": k, "Emails": ", ".join(v)} for k, v in groups.items()]
    st.dataframe(pd.DataFrame(group_rows), use_container_width=True)

    new_group_name = st.text_input("Tên nhóm mới", key="new_group_name")
    new_group_emails = st.text_area("Emails (mỗi email cách nhau dấu phẩy)", key="new_group_emails")
    if st.button("Create / Update Group", use_container_width=True):
        if new_group_name.strip():
            groups[new_group_name.strip()] = [e.strip() for e in new_group_emails.replace("\n", ",").split(",") if e.strip()]
            save_recipient_groups(groups)
            log_audit(st.session_state["current_user"]["username"], "save_group", new_group_name.strip())
            st.success("Đã lưu nhóm.")
            st.rerun()

    if groups:
        selected_group = st.selectbox("Chọn nhóm để sửa", list(groups.keys()), key="selected_group")
        edit_emails = st.text_area("Emails của nhóm", value=", ".join(groups.get(selected_group, [])), key="edit_group_emails")
        if st.button("Save Selected Group", use_container_width=True):
            groups[selected_group] = [e.strip() for e in edit_emails.replace("\n", ",").split(",") if e.strip()]
            save_recipient_groups(groups)
            log_audit(st.session_state["current_user"]["username"], "update_group", selected_group)
            st.success("Đã cập nhật nhóm.")
            st.rerun()


def admin_logs_tab():
    st.markdown("### Audit Logs")
    if AUDIT_LOG_FILE.exists():
        st.dataframe(pd.read_csv(AUDIT_LOG_FILE), use_container_width=True)
    st.markdown("### Email Logs")
    if EMAIL_LOG_FILE.exists():
        st.dataframe(pd.read_csv(EMAIL_LOG_FILE), use_container_width=True)


def admin_schedule_tab(cfg: dict):
    st.markdown("### Scheduler 09:00")
    auto_send_enabled = st.checkbox("Bật auto gửi email", value=cfg.get("auto_send_enabled", False), key="auto_send_enabled")
    auto_send_time = st.text_input("Giờ gửi tự động", value=cfg.get("auto_send_time", "09:00"), key="auto_send_time")
    groups = load_recipient_groups()
    current_groups = cfg.get("auto_send_groups", ["internal_morning"])
    auto_groups = st.multiselect("Nhóm nhận mail tự động", list(groups.keys()), default=[g for g in current_groups if g in groups], key="auto_groups")
    if st.button("Save Schedule", use_container_width=True):
        cfg["auto_send_enabled"] = auto_send_enabled
        cfg["auto_send_time"] = auto_send_time
        cfg["auto_send_groups"] = auto_groups
        save_config(cfg)
        log_audit(st.session_state["current_user"]["username"], "save_schedule", f"{auto_send_time} | {auto_groups}")
        st.success("Đã lưu scheduler.")


def main_app():
    ensure_files()
    init_session()
    start_scheduler_if_needed()
    cfg = load_config()

    current_user = st.session_state["current_user"]
    api_key = get_secret("OPENAI_API_KEY", "")
    sender = get_secret("SENDER_EMAIL", "")
    sender_pw = get_secret("SENDER_PASSWORD", "")
    recipients_default = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(f"Xin chào {current_user.get('full_name') or current_user.get('username')} | role: {current_user.get('role')}")

    ctop1, ctop2 = st.columns([8, 1])
    with ctop2:
        if st.button("Logout"):
            log_audit(current_user.get("username", ""), "logout")
            st.session_state["current_user"] = None
            st.rerun()

    st.sidebar.title("⚙️ Cấu hình")
    model = st.sidebar.text_input("Model", value=cfg.get("model", DEFAULT_MODEL))
    report_date = str(st.sidebar.date_input("Ngày báo cáo"))
    house_view = st.sidebar.selectbox("House View", ["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"],
                                      index=["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"].index(cfg.get("house_view", "Neutral")))
    smart_mode = st.sidebar.checkbox("Smart mode", value=cfg.get("smart_mode", True))
    recipients_manual = st.sidebar.text_area("Recipients thủ công", value=recipients_default, height=80)
    trade_ideas_count = st.sidebar.number_input("Trade ideas count", min_value=3, max_value=10, value=int(cfg.get("trade_ideas_count", 7)))
    auto_send_after_run_all = st.sidebar.checkbox("Run All xong tự gửi email", value=cfg.get("auto_send_after_run_all", False))

    if st.sidebar.button("💾 Lưu config"):
        cfg["model"] = model
        cfg["house_view"] = house_view
        cfg["smart_mode"] = smart_mode
        cfg["default_recipients"] = recipients_manual
        cfg["trade_ideas_count"] = int(trade_ideas_count)
        cfg["auto_send_after_run_all"] = auto_send_after_run_all
        save_config(cfg)
        log_audit(current_user["username"], "save_config")
        st.sidebar.success("Đã lưu config.")

    st.session_state["user_news"] = st.text_area("Tin bổ sung / Additional notes", value=st.session_state.get("user_news", ""), height=100)
    st.session_state["expert_notes"] = st.text_area("Khuyến nghị chuyên gia / quỹ / Expert & fund recommendations", value=st.session_state.get("expert_notes", ""), height=140)

    if has_permission("run_pipeline"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Refresh Market", use_container_width=True):
                st.session_state["market_df"] = fetch_market_snapshot(DEFAULT_TICKERS)
        with c2:
            if st.button("📰 Load News", use_container_width=True):
                news_df = fetch_rss_news(RSS_FEEDS, max_per_feed=6)
                st.session_state["news_df"] = news_df
                st.session_state["news_df_ai"] = filter_news_for_ai(news_df, smart_mode=smart_mode)
        with c3:
            if st.button("🚀 Run All", use_container_width=True):
                result = run_all_pipeline(api_key, model, report_date, cfg, st.session_state["user_news"], st.session_state["expert_notes"])
                for k, v in result.items():
                    st.session_state[k] = v

                save_market_history(report_date, st.session_state["morning_note"], st.session_state["closing_note"])
                save_ic_history(report_date, st.session_state["ic_note"])
                save_trade_ideas_history(report_date, st.session_state["trade_ideas"])
                log_audit(current_user["username"], "run_all")

                if cfg.get("auto_send_after_run_all", False) and has_permission("send_email"):
                    recipients_list = [x.strip() for x in recipients_manual.replace(";", ",").split(",") if x.strip()]
                    ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
                    log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)

                st.success("Đã chạy xong toàn bộ pipeline.")

    tab_names = ["Dashboard", "Notes", "Allocation", "Trade Ideas", "Email"]
    if has_permission("manage_users") or has_permission("manage_groups") or has_permission("view_logs") or has_permission("manage_schedule"):
        tab_names += ["Users", "Recipient Groups", "Logs", "Scheduler"]

    tabs = st.tabs(tab_names)
    tab_map = {name: tabs[i] for i, name in enumerate(tab_names)}

    with tab_map["Dashboard"]:
        if not st.session_state["market_df"].empty:
            st.subheader("Market")
            st.dataframe(st.session_state["market_df"], use_container_width=True)
            st.text_area("Market Highlights", build_market_highlights(st.session_state["market_df"]), height=130)
            st.write(f"**Market Bias:** {get_market_bias(st.session_state['market_df'])}")

            st.markdown("### Vietnam Market Focus")
            vn_market = st.session_state["market_df"][st.session_state["market_df"]["Asset"].isin(["VNINDEX", "USD/VND (proxy)"])]
            if not vn_market.empty:
                st.dataframe(vn_market, use_container_width=True)

            st.markdown("### Recommended Vietnam Watchlist")
            vn_watch_df = fetch_vn_recommendation_watchlist()
            if not vn_watch_df.empty:
                st.dataframe(vn_watch_df, use_container_width=True)

        if not st.session_state["news_df"].empty:
            vn_df, global_df = split_news_by_region(st.session_state["news_df"])
            a, b = st.columns(2)
            with a:
                st.markdown("#### Vietnam News")
                if not vn_df.empty:
                    vn_show = vn_df.copy()
                    vn_show["ArticleLink"] = vn_show["Link"].apply(lambda x: f'<a href="{x}" target="_blank">Mở bài</a>' if pd.notna(x) and str(x).strip() else "")
                    show_cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in vn_show.columns]
                    st.write(vn_show[show_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
            with b:
                st.markdown("#### Global News")
                if not global_df.empty:
                    global_show = global_df.copy()
                    global_show["ArticleLink"] = global_show["Link"].apply(lambda x: f'<a href="{x}" target="_blank">Open article</a>' if pd.notna(x) and str(x).strip() else "")
                    show_cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in global_show.columns]
                    st.write(global_show[show_cols].to_html(escape=False, index=False), unsafe_allow_html=True)

            st.text_area("Signals", build_top_actionable_signals(st.session_state["news_df_ai"], 5), height=120)

    with tab_map["Notes"]:
        if has_permission("generate_notes"):
            a, b, c = st.columns(3)
            with a:
                if st.button("Generate Morning Note", use_container_width=True):
                    allocation_text = st.session_state.get("allocation_text", "")
                    if not allocation_text and not st.session_state["market_df"].empty:
                        st.session_state["allocation_text"] = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                        allocation_text = st.session_state["allocation_text"]
                    st.session_state["morning_note"] = generate_morning_note(api_key, model, report_date, house_view, st.session_state["market_df"], st.session_state["news_df_ai"], st.session_state["user_news"], st.session_state["expert_notes"], allocation_text)

            with b:
                if st.button("Generate Closing Note", use_container_width=True):
                    allocation_text = st.session_state.get("allocation_text", "")
                    if not allocation_text and not st.session_state["market_df"].empty:
                        st.session_state["allocation_text"] = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                        allocation_text = st.session_state["allocation_text"]
                    st.session_state["closing_note"] = generate_closing_note(api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"], st.session_state["expert_notes"], allocation_text)

            with c:
                if st.button("Generate IC Note", use_container_width=True):
                    allocation_text = st.session_state.get("allocation_text", "")
                    if not allocation_text and not st.session_state["market_df"].empty:
                        st.session_state["allocation_text"] = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                        allocation_text = st.session_state["allocation_text"]
                    st.session_state["ic_note"] = generate_ic_note(api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"], st.session_state["expert_notes"], allocation_text)

        if st.session_state["morning_note"]:
            st.text_area("Morning Note", st.session_state["morning_note"], height=320)
            if has_permission("export_notes"):
                render_export_buttons("Morning Note", st.session_state["morning_note"], f"Morning_Note_{report_date}")

        if st.session_state["closing_note"]:
            st.text_area("Closing Note", st.session_state["closing_note"], height=320)
            if has_permission("export_notes"):
                render_export_buttons("Closing Note", st.session_state["closing_note"], f"Closing_Note_{report_date}")

        if st.session_state["ic_note"]:
            st.text_area("IC Note", st.session_state["ic_note"], height=380)
            if has_permission("export_notes"):
                render_export_buttons("IC Note", st.session_state["ic_note"], f"IC_Note_{report_date}")

    with tab_map["Allocation"]:
        st.markdown("### Portfolio Allocation Engine")
        if st.button("Compute Allocation", use_container_width=True):
            allocation = compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"])
            st.session_state["allocation_text"] = format_allocation(allocation)
        if st.session_state["allocation_text"]:
            st.text_area("Allocation Output", st.session_state["allocation_text"], height=280)

    with tab_map["Trade Ideas"]:
        if st.button("Generate Ranked Trade Ideas", use_container_width=True):
            st.session_state["trade_ideas_df"] = generate_ranked_trade_ideas(st.session_state["market_df"], st.session_state["news_df_ai"], int(trade_ideas_count))
            st.session_state["trade_ideas"] = format_trade_ideas_df(st.session_state["trade_ideas_df"])

        if isinstance(st.session_state.get("trade_ideas_df"), pd.DataFrame) and not st.session_state["trade_ideas_df"].empty:
            st.dataframe(st.session_state["trade_ideas_df"], use_container_width=True)

        if st.session_state["trade_ideas"]:
            st.text_area("Trade Ideas", st.session_state["trade_ideas"], height=320)
            st.text_area("Top Idea", extract_top_idea(st.session_state["trade_ideas"]), height=120)

    with tab_map["Email"]:
        groups = load_recipient_groups()
        selected_groups = st.multiselect("Chọn nhóm người nhận", list(groups.keys()), default=[])
        base_note = st.session_state["morning_note"] or st.session_state["closing_note"] or ""

        if st.button("Generate Email", use_container_width=True) and base_note:
            subject, body = build_email_bundle(api_key, model, report_date, "MANUAL", base_note, st.session_state.get("ic_note", ""), st.session_state.get("trade_ideas", ""), st.session_state.get("allocation_text", ""))
            st.session_state["email_subject"] = subject
            st.session_state["email_body"] = body

        st.session_state["email_subject"] = st.text_input("Subject", value=st.session_state.get("email_subject", ""))
        st.session_state["email_body"] = st.text_area("Body", value=st.session_state.get("email_body", ""), height=320)

        if has_permission("send_email"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Send Email Manual List", use_container_width=True):
                    recipients_list = [x.strip() for x in recipients_manual.replace(";", ",").split(",") if x.strip()]
                    ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
                    log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)
                    if ok:
                        st.success("Đã gửi email.")
            with c2:
                if st.button("Send to Selected Groups", use_container_width=True):
                    recipients_list = get_emails_from_groups(selected_groups)
                    ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
                    log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)
                    if ok:
                        st.success("Đã gửi email theo nhóm.")

    if "Users" in tab_map:
        with tab_map["Users"]:
            if has_permission("manage_users"):
                admin_users_tab()

    if "Recipient Groups" in tab_map:
        with tab_map["Recipient Groups"]:
            if has_permission("manage_groups"):
                admin_groups_tab()

    if "Logs" in tab_map:
        with tab_map["Logs"]:
            if has_permission("view_logs"):
                admin_logs_tab()

    if "Scheduler" in tab_map:
        with tab_map["Scheduler"]:
            if has_permission("manage_schedule"):
                admin_schedule_tab(cfg)


def main():
    ensure_files()
    init_session()
    if st.session_state.get("current_user") is None:
        login_screen()
    else:
        main_app()


if __name__ == "__main__":
    if "--auto" in sys.argv:
        scheduled_morning_job()
    else:
        main()
