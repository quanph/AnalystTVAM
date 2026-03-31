from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    A4 = None
    canvas = None

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None

try:
    import bcrypt
except Exception:
    bcrypt = None


APP_TITLE = "📊 Analyst Dashboard TVAM@2026"
DEFAULT_MODEL = "gpt-4.1-mini"

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"
CACHE_FILE = DATA_DIR / "ai_cache.json"
USERS_FILE = DATA_DIR / "users.json"
RECIPIENT_GROUPS_FILE = DATA_DIR / "recipient_groups.json"

EMAIL_LOG_FILE = DATA_DIR / "email_send_log.csv"
AUDIT_LOG_FILE = DATA_DIR / "audit_log.csv"
MARKET_HISTORY_FILE = DATA_DIR / "market_history.csv"
IC_HISTORY_FILE = DATA_DIR / "ic_note_history.csv"
TRADE_IDEAS_HISTORY_FILE = DATA_DIR / "trade_ideas_history.csv"
TARGET_PRICE_HISTORY_FILE = DATA_DIR / "target_price_history.csv"
SECTOR_HEATMAP_HISTORY_FILE = DATA_DIR / "sector_heatmap_history.csv"
PORTFOLIO_MEMO_HISTORY_FILE = DATA_DIR / "portfolio_memo_history.csv"
ACTION_SIGNAL_HISTORY_FILE = DATA_DIR / "action_signal_history.csv"
MODEL_PORTFOLIO_HISTORY_FILE = DATA_DIR / "model_portfolio_history.csv"

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

VN_EXPERT_RECOMMENDATIONS = [
    {"ticker": "FPT", "action": "Buy", "source": "SSI"},
    {"ticker": "FPT", "action": "Buy", "source": "VNDirect"},
    {"ticker": "FPT", "action": "Buy", "source": "Fund"},
    {"ticker": "VCB", "action": "Buy", "source": "Broker"},
    {"ticker": "VCB", "action": "Buy", "source": "Fund"},
    {"ticker": "MBB", "action": "Buy", "source": "Fund"},
    {"ticker": "HPG", "action": "Watch", "source": "Expert"},
    {"ticker": "SSI", "action": "Watch", "source": "Broker"},
    {"ticker": "MWG", "action": "Buy", "source": "Retail Research"},
    {"ticker": "ACB", "action": "Buy", "source": "Research"},
]

VN_SECTOR_MAP = {
    "FPT": "Technology",
    "VCB": "Banks",
    "MBB": "Banks",
    "ACB": "Banks",
    "TCB": "Banks",
    "HPG": "Steel",
    "MWG": "Consumer",
    "SSI": "Securities",
    "VHM": "Real Estate",
    "GAS": "Energy",
}

MODEL_PORTFOLIO_V65 = {
    "Core": [
        {"ticker": "FPT.VN", "weight": 0.20},
        {"ticker": "VCB.VN", "weight": 0.20},
        {"ticker": "MBB.VN", "weight": 0.15},
    ],
    "Tactical": [
        {"ticker": "MWG.VN", "weight": 0.10},
        {"ticker": "SSI.VN", "weight": 0.10},
        {"ticker": "HPG.VN", "weight": 0.10},
    ],
    "Hedge": [
        {"ticker": "Cash", "weight": 0.10},
        {"ticker": "USD", "weight": 0.05},
    ]
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

SCHEDULER_STARTED = False


# =========================================================
# FILE / JSON / LOG
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


def append_csv_row(path: Path, row: dict) -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def log_audit(username: str, action: str, detail: str = "") -> None:
    append_csv_row(AUDIT_LOG_FILE, {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username": username,
        "Action": action,
        "Detail": detail,
    })


def ensure_files() -> None:
    if not CONFIG_FILE.exists():
        save_json(CONFIG_FILE, {
            "model": DEFAULT_MODEL,
            "house_view": "Neutral",
            "default_recipients": "",
            "smart_mode": True,
            "auto_send_after_run_all": False,
            "trade_ideas_count": 8,
            "auto_send_enabled": False,
            "auto_send_time": "09:00",
            "auto_send_groups": ["internal_morning"],
            "auto_user_news": "",
            "auto_expert_notes": ""
        })

    if not CACHE_FILE.exists():
        save_json(CACHE_FILE, {})

    if not USERS_FILE.exists():
        save_json(USERS_FILE, [{
            "username": "admin",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "full_name": "System Admin",
            "active": True
        }])

    if not RECIPIENT_GROUPS_FILE.exists():
        save_json(RECIPIENT_GROUPS_FILE, {
            "internal_morning": [],
            "cio_team": [],
            "pm_team": [],
            "clients_vip": []
        })

    init_csv(EMAIL_LOG_FILE, ["SendTime", "Sender", "Recipients", "Subject", "Status", "Error"])
    init_csv(AUDIT_LOG_FILE, ["Time", "Username", "Action", "Detail"])
    init_csv(MARKET_HISTORY_FILE, ["Date", "MorningNote", "ClosingNote"])
    init_csv(IC_HISTORY_FILE, ["Date", "ICNote"])
    init_csv(TRADE_IDEAS_HISTORY_FILE, ["Date", "TradeIdeas"])
    init_csv(TARGET_PRICE_HISTORY_FILE, ["Date", "Ticker", "Direction", "TargetZone", "Conviction", "WhyNow"])
    init_csv(SECTOR_HEATMAP_HISTORY_FILE, ["Date", "Sector", "IdeaCount", "AvgScore", "AvgConviction", "Heat"])
    init_csv(PORTFOLIO_MEMO_HISTORY_FILE, ["Date", "PortfolioMemo"])
    init_csv(ACTION_SIGNAL_HISTORY_FILE, ["Date", "Ticker", "Action", "Reason", "Score", "Conviction"])
    init_csv(MODEL_PORTFOLIO_HISTORY_FILE, ["Date", "Bucket", "Ticker", "Weight", "ReturnPct", "Contribution"])


def load_config() -> dict:
    return load_json(CONFIG_FILE, {})


def save_config(cfg: dict) -> None:
    save_json(CONFIG_FILE, cfg)


def load_cache() -> dict:
    return load_json(CACHE_FILE, {})


def save_cache(cache: dict) -> None:
    save_json(CACHE_FILE, cache)


# =========================================================
# AUTH / USERS / GROUPS
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


def authenticate(username: str, password: str) -> Optional[dict]:
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
def ai_is_available(api_key: str) -> bool:
    return bool(api_key) and (OpenAI is not None)


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


def fetch_last_price_for_ticker(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if hist.empty:
            return None
        return float(hist.iloc[-1]["Close"])
    except Exception:
        return None


def fetch_day_return_for_ticker(ticker: str) -> float:
    if ticker == "Cash":
        return 0.0
    if ticker == "USD":
        return 0.0
    try:
        hist = yf.Ticker(ticker).history(period="2d", interval="1d", auto_adjust=False)
        if hist.empty or len(hist) < 2:
            return 0.0
        last_close = float(hist.iloc[-1]["Close"])
        prev_close = float(hist.iloc[-2]["Close"])
        if prev_close == 0:
            return 0.0
        return round((last_close - prev_close) / prev_close * 100, 2)
    except Exception:
        return 0.0


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
    for k in ["vnindex", "vietnam", "sbv", "usd", "oil", "bank", "fpt", "vcb", "hpg", "mbb"]:
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


def _get_change(market_df: pd.DataFrame, asset_name: str) -> float:
    row = market_df.loc[market_df["Asset"] == asset_name]
    if row.empty or pd.isna(row.iloc[0]["ChangePct"]):
        return 0.0
    return float(row.iloc[0]["ChangePct"])


# =========================================================
# CONSENSUS / SECTOR
# =========================================================
def summarize_vn_expert_recommendations(recs: List[dict]) -> pd.DataFrame:
    if not recs:
        return pd.DataFrame(columns=["Ticker", "Mentions", "Buy", "Watch", "Sell", "Consensus", "Sources"])
    df = pd.DataFrame(recs)
    rows = []
    for ticker, grp in df.groupby("ticker"):
        buy = int((grp["action"].str.lower() == "buy").sum())
        watch = int((grp["action"].str.lower() == "watch").sum())
        sell = int((grp["action"].str.lower().isin(["sell", "reduce", "underweight"])).sum())
        mentions = len(grp)
        sources = ", ".join(sorted(grp["source"].dropna().astype(str).unique().tolist()))
        if buy >= max(watch, sell):
            consensus = "Buy / Mua"
        elif sell > max(buy, watch):
            consensus = "Reduce / Giảm tỷ trọng"
        else:
            consensus = "Watch / Theo dõi"
        rows.append({
            "Ticker": ticker,
            "Mentions": mentions,
            "Buy": buy,
            "Watch": watch,
            "Sell": sell,
            "Consensus": consensus,
            "Sources": sources
        })
    return pd.DataFrame(rows).sort_values(by=["Mentions", "Buy", "Watch"], ascending=[False, False, False]).reset_index(drop=True)


def build_vn_consensus_text(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No VN expert consensus / Chưa có đồng thuận chuyên gia Việt Nam."
    lines = ["VIETNAM STOCK CONSENSUS / CỔ PHIẾU VIỆT NAM ĐƯỢC ĐỀ XUẤT"]
    for _, r in df.head(8).iterrows():
        lines.append(
            f"- {r['Ticker']}: {r['Consensus']} | Mentions={r['Mentions']} | "
            f"Buy={r['Buy']} | Watch={r['Watch']} | Sell={r['Sell']} | Sources={r['Sources']}"
        )
    return "\n".join(lines)


def conviction_to_num(x: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(x), 1)


def build_sector_conviction_heatmap(trade_ideas_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    if trade_ideas_df.empty:
        return pd.DataFrame()

    df = trade_ideas_df.copy()
    df["Ticker"] = df["Asset"].astype(str).str.replace(".VN", "", regex=False).str.replace("Vietnam Banks", "VCB", regex=False)
    df["Sector"] = df["Ticker"].map(sector_map).fillna("Other")
    df["ConvictionNum"] = df["Conviction"].map(conviction_to_num).fillna(1)

    out = df.groupby("Sector").agg(
        IdeaCount=("Ticker", "count"),
        AvgScore=("Score", "mean"),
        AvgConviction=("ConvictionNum", "mean")
    ).reset_index()

    def heat(x):
        if x >= 2.7:
            return "Very High"
        if x >= 2.2:
            return "High"
        if x >= 1.6:
            return "Medium"
        return "Low"

    out["Heat"] = out["AvgConviction"].apply(heat)
    out["AvgScore"] = out["AvgScore"].round(2)
    out["AvgConviction"] = out["AvgConviction"].round(2)
    return out.sort_values(by=["AvgConviction", "AvgScore"], ascending=False).reset_index(drop=True)


# =========================================================
# ALLOCATION V6.5
# =========================================================
def compute_portfolio_allocation_v65(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    consensus_df: pd.DataFrame | None = None
) -> dict:
    oil_chg = _get_change(market_df, "Oil (WTI)")
    gold_chg = _get_change(market_df, "Gold")
    usd_chg = _get_change(market_df, "USD Index")
    vn_chg = _get_change(market_df, "VNINDEX")
    spx_chg = _get_change(market_df, "S&P 500")
    ndx_chg = _get_change(market_df, "Nasdaq")
    us10y_chg = _get_change(market_df, "US 10Y Yield")
    usdvnd_chg = _get_change(market_df, "USD/VND (proxy)")

    regime_score = 0
    if spx_chg > 0.5:
        regime_score += 1
    if ndx_chg > 0.5:
        regime_score += 1
    if vn_chg > 0.7:
        regime_score += 1
    if usd_chg > 0.5:
        regime_score -= 1
    if oil_chg > 1.5:
        regime_score -= 1
    if us10y_chg > 0.5:
        regime_score -= 1

    if regime_score >= 2:
        regime = "Risk-on"
    elif regime_score <= -1:
        regime = "Risk-off"
    else:
        regime = "Transition"

    vn_consensus_strength = 0
    vn_top_tickers = []
    if consensus_df is not None and not consensus_df.empty:
        if "Buy" in consensus_df.columns and int(consensus_df["Buy"].sum()) >= 5:
            vn_consensus_strength = 1
        if "Ticker" in consensus_df.columns:
            vn_top_tickers = consensus_df.head(5)["Ticker"].astype(str).tolist()

    allocation = {
        "Regime": regime,
        "Global Equity": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "1-3 months",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Core risk asset",
        },
        "Vietnam Equity": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "1-3 months",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Core + tactical alpha",
        },
        "Global Fixed Income": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "2-6 weeks",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Hedge / duration stabilizer",
        },
        "Vietnam Fixed Income": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "1-3 months",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Liquidity anchor",
        },
        "Oil": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "2-6 weeks",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Inflation hedge",
        },
        "Gold": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "2-6 weeks",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Defensive hedge",
        },
        "USD": {
            "View": "Neutral",
            "Conviction": "Medium",
            "Horizon": "2-6 weeks",
            "Rationale": "",
            "Risk": "",
            "Trigger": "",
            "PortfolioFit": "Macro hedge",
        },
    }

    if spx_chg > 0.5 and ndx_chg > 0 and us10y_chg <= 0.3 and usd_chg <= 0.5:
        allocation["Global Equity"]["View"] = "Overweight"
        allocation["Global Equity"]["Rationale"] = (
            "Động lượng cổ phiếu toàn cầu còn tích cực khi lợi suất và USD chưa siết mạnh. "
            "Global equity momentum remains constructive while yields and USD are not tightening aggressively."
        )
        allocation["Global Equity"]["Risk"] = (
            "Rủi ro là lợi suất tăng lại hoặc USD mạnh lên. "
            "Main risk is a renewed yield spike or stronger USD."
        )
        allocation["Global Equity"]["Trigger"] = (
            "Tăng thêm khi S&P 500 và Nasdaq duy trì xu hướng tăng với yields ổn định. "
            "Add if S&P 500 and Nasdaq maintain positive momentum with stable yields."
        )
    elif usd_chg > 0.5 or us10y_chg > 0.5:
        allocation["Global Equity"]["View"] = "Underweight"
        allocation["Global Equity"]["Rationale"] = (
            "USD mạnh và yields cao gây áp lực lên định giá. "
            "Stronger USD and higher yields pressure valuation."
        )
        allocation["Global Equity"]["Risk"] = (
            "Có thể bỏ lỡ nhịp hồi nếu stress macro giảm nhanh. "
            "Could miss a rebound if macro stress fades quickly."
        )
        allocation["Global Equity"]["Trigger"] = (
            "Nâng lại khi yields ổn định và USD dịu xuống. "
            "Re-upgrade when yields stabilize and USD cools."
        )

    if vn_chg > 0.7:
        allocation["Vietnam Equity"]["View"] = "Overweight"
        allocation["Vietnam Equity"]["Conviction"] = "High" if vn_consensus_strength == 1 else "Medium"
        allocation["Vietnam Equity"]["Rationale"] = (
            f"Động lượng nội địa tích cực, thêm hỗ trợ từ đồng thuận chuyên gia cho các mã như {', '.join(vn_top_tickers) if vn_top_tickers else 'large caps'}. "
            f"Domestic momentum is constructive, with added support from expert consensus on names such as {', '.join(vn_top_tickers) if vn_top_tickers else 'large caps'}."
        )
        allocation["Vietnam Equity"]["Risk"] = (
            "Rủi ro là FX pressure, khối ngoại bán ròng hoặc nhóm dẫn dắt suy yếu. "
            "Risks are FX pressure, foreign outflows, or leadership breakdown."
        )
        allocation["Vietnam Equity"]["Trigger"] = (
            "Tăng thêm khi VNIndex và large caps tiếp tục dẫn dắt, USD/VND ổn định. "
            "Add if VNIndex and large caps continue to lead with stable USD/VND."
        )
    elif usdvnd_chg > 0.7:
        allocation["Vietnam Equity"]["View"] = "Neutral"
        allocation["Vietnam Equity"]["Rationale"] = (
            "Áp lực tỷ giá hạn chế risk appetite ngắn hạn. "
            "FX stress limits near-term risk appetite."
        )
        allocation["Vietnam Equity"]["Risk"] = (
            "USD/VND tăng tiếp sẽ gây áp lực thêm. "
            "Further USD/VND upside would add pressure."
        )
        allocation["Vietnam Equity"]["Trigger"] = (
            "Nâng lại khi USD/VND ổn định và dòng tiền quay lại large caps. "
            "Upgrade if USD/VND stabilizes and flows return to large caps."
        )
    else:
        allocation["Vietnam Equity"]["Rationale"] = (
            "Cần thêm xác nhận từ dòng tiền và nhóm dẫn dắt. "
            "Needs further confirmation from flows and leadership."
        )
        allocation["Vietnam Equity"]["Risk"] = (
            "Foreign outflows và FX pressure. "
            "Foreign outflows and FX pressure."
        )
        allocation["Vietnam Equity"]["Trigger"] = (
            "Nâng lên khi VNIndex bứt phá với thanh khoản cải thiện. "
            "Upgrade on a VNIndex breakout with improving liquidity."
        )

    if us10y_chg > 0.5 or oil_chg > 1.5:
        allocation["Global Fixed Income"]["View"] = "Underweight"
        allocation["Global Fixed Income"]["Rationale"] = (
            "Yields tăng và rủi ro lạm phát từ dầu khiến duration kém hấp dẫn. "
            "Rising yields and oil-driven inflation risk argue against adding duration."
        )
        allocation["Global Fixed Income"]["Risk"] = (
            "Growth scare bất ngờ có thể khiến duration bật tăng. "
            "A sudden growth scare could trigger a duration rally."
        )
        allocation["Global Fixed Income"]["Trigger"] = (
            "Nâng lên khi yields và dầu ổn định hơn. "
            "Upgrade when yields and oil stabilize."
        )
    elif us10y_chg < -0.3:
        allocation["Global Fixed Income"]["View"] = "Overweight"
        allocation["Global Fixed Income"]["Rationale"] = (
            "Lợi suất hạ hỗ trợ duration. "
            "Falling yields support duration."
        )
        allocation["Global Fixed Income"]["Risk"] = (
            "Inflation quay lại. "
            "Inflation re-accelerates."
        )
        allocation["Global Fixed Income"]["Trigger"] = (
            "Tăng thêm khi yields tiếp tục giảm. "
            "Add if yields continue to ease."
        )
    else:
        allocation["Global Fixed Income"]["Rationale"] = (
            "Chưa đủ tín hiệu để tăng duration mạnh. "
            "Not enough evidence yet to aggressively add duration."
        )
        allocation["Global Fixed Income"]["Risk"] = (
            "Inflation shock hoặc yield spike. "
            "Inflation shock or yield spike."
        )
        allocation["Global Fixed Income"]["Trigger"] = (
            "Nâng lên khi inflation risk dịu xuống. "
            "Upgrade when inflation risk fades."
        )

    allocation["Vietnam Fixed Income"]["Rationale"] = (
        "Theo dõi thanh khoản nội địa, FX và định hướng điều hành để đánh giá cơ hội duration trong nước. "
        "Monitor domestic liquidity, FX and policy direction to assess local duration opportunities."
    )
    allocation["Vietnam Fixed Income"]["Risk"] = (
        "Áp lực tỷ giá bất ngờ hoặc thanh khoản thắt lại. "
        "Unexpected FX pressure or tighter liquidity."
    )
    allocation["Vietnam Fixed Income"]["Trigger"] = (
        "Nâng lên khi FX ổn định và thanh khoản cải thiện. "
        "Upgrade when FX stabilizes and liquidity improves."
    )

    if oil_chg > 1:
        allocation["Oil"]["View"] = "Overweight"
        allocation["Oil"]["Rationale"] = (
            "Động lượng dầu tích cực và vai trò hedge lạm phát hỗ trợ tactical exposure. "
            "Positive oil momentum and its inflation-hedge role support tactical exposure."
        )
        allocation["Oil"]["Risk"] = (
            "Đảo chiều nhanh nếu căng thẳng dịu đi. "
            "Sharp reversal if geopolitical stress eases."
        )
        allocation["Oil"]["Trigger"] = (
            "Giảm tỷ trọng nếu oil momentum suy yếu. "
            "Reduce if oil momentum fades."
        )
    else:
        allocation["Oil"]["Rationale"] = (
            "Dầu phù hợp như tactical hedge hơn là core holding. "
            "Oil works better as a tactical hedge than a core holding."
        )
        allocation["Oil"]["Risk"] = (
            "Headline volatility. "
            "Headline volatility."
        )
        allocation["Oil"]["Trigger"] = (
            "Tăng khi có break-out rõ hơn. "
            "Add on a clearer breakout."
        )

    if gold_chg > 0.5 or regime == "Risk-off":
        allocation["Gold"]["View"] = "Overweight"
        allocation["Gold"]["Rationale"] = (
            "Vàng phù hợp như lớp phòng thủ trong môi trường volatile. "
            "Gold fits as a defensive layer in volatile conditions."
        )
        allocation["Gold"]["Risk"] = (
            "Real yields tăng nhanh có thể gây áp lực lên vàng. "
            "Rapidly rising real yields may pressure gold."
        )
        allocation["Gold"]["Trigger"] = (
            "Tăng thêm nếu risk-off kéo dài. "
            "Add if the risk-off regime persists."
        )
    else:
        allocation["Gold"]["Rationale"] = (
            "Vàng hữu ích như hedge khi uncertainty tăng lên. "
            "Gold is useful as a hedge when uncertainty rises."
        )
        allocation["Gold"]["Risk"] = (
            "Strong risk-on rebound may reduce hedge demand. "
            "Strong risk-on rebound may reduce hedge demand."
        )
        allocation["Gold"]["Trigger"] = (
            "Nâng lên khi USD và volatility cùng tăng. "
            "Upgrade when both USD and volatility rise."
        )

    if usd_chg > 0.5:
        allocation["USD"]["View"] = "Long"
        allocation["USD"]["Rationale"] = (
            "USD mạnh hỗ trợ vai trò hedge trong môi trường biến động. "
            "USD strength supports its role as a hedge in volatile conditions."
        )
        allocation["USD"]["Risk"] = (
            "Risk sentiment hồi phục nhanh có thể đảo chiều USD. "
            "A quick rebound in risk sentiment may reverse USD strength."
        )
        allocation["USD"]["Trigger"] = (
            "Duy trì khi DXY còn xu hướng tăng. "
            "Maintain while DXY trend remains firm."
        )
    else:
        allocation["USD"]["Rationale"] = (
            "USD nên giữ trung lập nếu chưa có tín hiệu risk-off rõ. "
            "USD should remain neutral without a clear risk-off signal."
        )
        allocation["USD"]["Risk"] = (
            "Policy surprise hoặc sudden risk-off. "
            "Policy surprise or sudden risk-off."
        )
        allocation["USD"]["Trigger"] = (
            "Long khi DXY tăng rõ hơn. "
            "Shift to long if DXY strengthens more clearly."
        )

    return allocation


def format_allocation_v65(allocation: dict) -> str:
    lines = [
        "PORTFOLIO ALLOCATION / PHÂN BỔ DANH MỤC",
        f"- Market Regime / Chế độ thị trường: {allocation.get('Regime', 'Transition')}"
    ]
    for key, value in allocation.items():
        if key == "Regime":
            continue
        lines.append("")
        lines.append(f"{key}:")
        lines.append(f"- View: {value.get('View', 'Neutral')}")
        lines.append(f"- Conviction: {value.get('Conviction', 'Medium')}")
        lines.append(f"- Horizon: {value.get('Horizon', '1-3 months')}")
        lines.append(f"- Rationale: {value.get('Rationale', '')}")
        lines.append(f"- Risk: {value.get('Risk', '')}")
        lines.append(f"- Trigger: {value.get('Trigger', '')}")
        lines.append(f"- Portfolio Fit: {value.get('PortfolioFit', '')}")
    return "\n".join(lines)


# =========================================================
# TRADE IDEAS V6.5
# =========================================================
def _infer_target_zone(asset: str, market_df: pd.DataFrame) -> str:
    asset_upper = asset.upper()
    if asset_upper == "VNINDEX":
        price_row = market_df.loc[market_df["Asset"] == "VNINDEX"]
        if not price_row.empty:
            px = float(price_row.iloc[0]["Price"])
            low = round(px * 1.02, 1)
            high = round(px * 1.05, 1)
            return f"{low} - {high}"
        return "Breakout zone"
    if ".VN" in asset_upper:
        ticker = asset_upper
        last_price = fetch_last_price_for_ticker(ticker)
        if last_price:
            low = round(last_price * 1.05, 2)
            high = round(last_price * 1.12, 2)
            return f"{low} - {high}"
        return "+5% đến +12%"
    if asset_upper == "OIL":
        return "+4% đến +8%"
    if asset_upper == "GOLD":
        return "+3% đến +6%"
    if asset_upper == "USD":
        return "DXY +1% đến +3%"
    if "BANK" in asset_upper:
        return "+6% đến +10%"
    return "Tactical upside zone"


def score_trade_idea_v65(asset: str, market_df: pd.DataFrame, news_df: pd.DataFrame, consensus_df: pd.DataFrame | None = None) -> dict:
    score = 5.0
    rationale = []

    oil_chg = _get_change(market_df, "Oil (WTI)")
    gold_chg = _get_change(market_df, "Gold")
    usd_chg = _get_change(market_df, "USD Index")
    vn_chg = _get_change(market_df, "VNINDEX")
    spx_chg = _get_change(market_df, "S&P 500")

    asset_lower = asset.lower()

    if "oil" in asset_lower:
        if oil_chg > 1:
            score += 2
            rationale.append("Oil momentum positive / Động lượng dầu tích cực")
        elif oil_chg < -1:
            score -= 1

    if "gold" in asset_lower:
        if gold_chg > 0.5 or usd_chg > 0.4:
            score += 1
            rationale.append("Defensive demand supports gold / Nhu cầu phòng thủ hỗ trợ vàng")

    if "usd" in asset_lower:
        if usd_chg > 0.5:
            score += 2
            rationale.append("USD strength confirmed / USD đang mạnh lên")

    if "vnindex" in asset_lower or ".vn" in asset_lower or "vietnam" in asset_lower:
        if vn_chg > 0.7:
            score += 1.5
            rationale.append("Vietnam momentum supportive / Động lượng Việt Nam tích cực")
        elif vn_chg < -0.8:
            score -= 1

    if "global equity" in asset_lower:
        if spx_chg > 0.5:
            score += 1.5
            rationale.append("Global equity momentum supportive / Động lượng toàn cầu tích cực")

    if not news_df.empty:
        matches = news_df[
            news_df["Title"].str.contains(asset.split()[0], case=False, na=False) |
            news_df["Summary"].str.contains(asset.split()[0], case=False, na=False)
        ]
        score += min(len(matches) * 0.5, 2.0)
        if len(matches) > 0:
            rationale.append(f"News flow support={len(matches)} / Có hỗ trợ từ news flow={len(matches)}")

    conviction = "Low"
    if score >= 8:
        conviction = "High"
    elif score >= 6:
        conviction = "Medium"

    portfolio_fit = "Tactical"
    why_now = "Cross-asset setup is supportive / Bối cảnh đa tài sản đang ủng hộ"

    clean_asset = asset.replace(".VN", "").replace("Vietnam ", "").replace(" Banks", "").upper()
    if consensus_df is not None and not consensus_df.empty and "Ticker" in consensus_df.columns:
        matched = consensus_df[consensus_df["Ticker"].astype(str).str.upper() == clean_asset]
        if not matched.empty:
            score += 0.5
            why_now = "Repeated expert mentions increase conviction / Nhiều chuyên gia cùng nhắc giúp tăng conviction"
            portfolio_fit = "Alpha / expert consensus"
            if score >= 7:
                conviction = "High"

    if asset.upper() in ["USD", "GOLD"]:
        portfolio_fit = "Hedge"
    elif asset.upper() in ["VNINDEX", "GLOBAL EQUITY", "VIETNAM BANKS"]:
        portfolio_fit = "Core + tactical"

    score = max(1.0, min(score, 10.0))
    if score >= 8.5:
        direction = "Buy"
    elif score >= 7:
        direction = "Add"
    elif score >= 5:
        direction = "Hold"
    elif score >= 4:
        direction = "Trim"
    else:
        direction = "Exit"

    horizon = "1-4 weeks" if score >= 7 else "Short-term watch"
    target_zone = _infer_target_zone(asset, market_df)

    return {
        "Asset": asset,
        "Action": direction,
        "Score": round(score, 1),
        "Conviction": conviction,
        "Confidence": conviction,
        "WhyNow": why_now,
        "PortfolioFit": portfolio_fit,
        "Horizon": horizon,
        "TargetZone": target_zone,
        "Rationale": "; ".join(rationale) if rationale else "Neutral setup / Thiết lập trung tính"
    }


def generate_ranked_trade_ideas_v65(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    consensus_df: pd.DataFrame | None = None,
    ideas_count: int = 8
) -> pd.DataFrame:
    universe = [
        "Oil", "Gold", "USD", "VNINDEX", "Vietnam Banks",
        "FPT.VN", "HPG.VN", "VCB.VN", "MBB.VN", "SSI.VN", "MWG.VN", "Global Equity"
    ]
    scored = [score_trade_idea_v65(asset, market_df, news_df, consensus_df) for asset in universe]
    df = pd.DataFrame(scored).sort_values(by=["Score", "Conviction"], ascending=[False, False]).head(ideas_count).reset_index(drop=True)
    df.index = df.index + 1
    return df


def format_trade_ideas_v65(df: pd.DataFrame, consensus_text: str = "") -> str:
    if df.empty:
        return "Chưa có trade ideas / No trade ideas."
    lines = ["TRADE IDEAS V6.5 / Ý TƯỞNG ĐẦU TƯ V6.5"]
    for idx, row in df.iterrows():
        lines.append(
            f"\n#{idx} {row['Asset']}\n"
            f"- Action / Lệnh gợi ý: {row['Action']}\n"
            f"- Score: {row['Score']}\n"
            f"- Conviction: {row['Conviction']}\n"
            f"- Why now / Vì sao lúc này: {row['WhyNow']}\n"
            f"- Portfolio fit / Vai trò trong danh mục: {row['PortfolioFit']}\n"
            f"- Horizon / Khung thời gian: {row['Horizon']}\n"
            f"- Target price / target zone giả lập: {row['TargetZone']}\n"
            f"- Rationale / Luận điểm: {row['Rationale']}"
        )
    if consensus_text:
        lines.append("")
        lines.append(consensus_text)
    return "\n".join(lines)


def extract_top_idea(trade_ideas_text: str) -> str:
    text = normalize_text(trade_ideas_text)
    if not text:
        return "Chưa có ý tưởng nổi bật / No top idea available."
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        return blocks[1]
    return "\n".join(text.splitlines()[:10])


# =========================================================
# MODEL PORTFOLIO / BENCHMARK / ACTIONS / MEMO
# =========================================================
def flatten_model_portfolio(model_portfolio: dict) -> List[dict]:
    rows = []
    for bucket, items in model_portfolio.items():
        for item in items:
            rows.append({
                "Bucket": bucket,
                "Ticker": item["ticker"],
                "Weight": float(item["weight"])
            })
    return rows


def compute_model_portfolio_benchmark(model_portfolio: dict, market_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    vnindex_row = market_df.loc[market_df["Asset"] == "VNINDEX"]
    vnindex_ret = float(vnindex_row.iloc[0]["ChangePct"]) if not vnindex_row.empty else 0.0

    portfolio_ret = 0.0
    for item in flatten_model_portfolio(model_portfolio):
        ticker = item["Ticker"]
        weight = item["Weight"]

        if ticker == "Cash":
            ret = 0.0
        elif ticker == "USD":
            ret = _get_change(market_df, "USD Index")
        else:
            ret = fetch_day_return_for_ticker(ticker)

        contribution = weight * ret
        portfolio_ret += contribution
        rows.append({
            "Bucket": item["Bucket"],
            "Ticker": ticker,
            "Weight": weight,
            "ReturnPct": ret,
            "Contribution": round(contribution, 2)
        })

    df = pd.DataFrame(rows)
    df.attrs["portfolio_return"] = round(portfolio_ret, 2)
    df.attrs["vnindex_return"] = round(vnindex_ret, 2)
    df.attrs["excess_return"] = round(portfolio_ret - vnindex_ret, 2)
    return df


def generate_action_signals(trade_ideas_df: pd.DataFrame, model_portfolio: dict) -> pd.DataFrame:
    if trade_ideas_df.empty:
        return pd.DataFrame()

    model_tickers = [x["Ticker"] for x in flatten_model_portfolio(model_portfolio)]
    rows = []

    for _, r in trade_ideas_df.iterrows():
        ticker = str(r["Asset"]).replace(".VN", "")
        score = float(r["Score"])
        conviction = str(r["Conviction"])
        action = str(r["Action"])
        in_portfolio = f"{ticker}.VN" in model_tickers or ticker in model_tickers

        if action == "Buy" and not in_portfolio:
            reason = "High conviction, strong setup, not yet in portfolio"
        elif action in ["Buy", "Add"] and in_portfolio:
            action = "Add"
            reason = "Position already exists and setup remains constructive"
        elif action == "Hold" and in_portfolio:
            action = "Hold"
            reason = "No major change in setup"
        elif action == "Trim":
            reason = "Score weakens and conviction softens"
        elif action == "Exit":
            reason = "Low score and weak setup"
        else:
            reason = "Monitor for better entry"

        rows.append({
            "Ticker": ticker,
            "Action": action,
            "Reason": reason,
            "Score": score,
            "Conviction": conviction
        })

    return pd.DataFrame(rows)


def build_portfolio_memo(allocation_text: str, trade_ideas_df: pd.DataFrame, action_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
    top_names = ", ".join(trade_ideas_df.head(3)["Asset"].astype(str).tolist()) if not trade_ideas_df.empty else "N/A"
    actions = ", ".join([f"{r['Ticker']}:{r['Action']}" for _, r in action_df.head(4).iterrows()]) if not action_df.empty else "No action"
    regime_line = allocation_text.splitlines()[1] if allocation_text and len(allocation_text.splitlines()) > 1 else "Regime: N/A"

    port_ret = benchmark_df.attrs.get("portfolio_return", 0.0) if benchmark_df is not None else 0.0
    vn_ret = benchmark_df.attrs.get("vnindex_return", 0.0) if benchmark_df is not None else 0.0
    excess = benchmark_df.attrs.get("excess_return", 0.0) if benchmark_df is not None else 0.0

    return f"""PORTFOLIO MEMO / GHI CHÚ DANH MỤC

{regime_line}

Benchmark:
- Model portfolio return: {port_ret}%
- VNINDEX return: {vn_ret}%
- Excess return: {excess}%

Top conviction:
- {top_names}

Action focus:
- {actions}

Main risk:
- Higher USD, yield volatility, and weaker market breadth.
- USD mạnh hơn, biến động lợi suất, và độ rộng thị trường suy yếu.

Read-through:
- Keep core exposure selective, use tactical adds only where conviction remains high.
- Giữ core exposure có chọn lọc, chỉ tăng tactical ở các mã còn conviction cao.
"""


# =========================================================
# TARGET TRACKING / HISTORY
# =========================================================
def save_target_price_history(report_date: str, trade_ideas_df: pd.DataFrame) -> None:
    if trade_ideas_df.empty:
        return
    for _, r in trade_ideas_df.iterrows():
        append_csv_row(TARGET_PRICE_HISTORY_FILE, {
            "Date": report_date,
            "Ticker": str(r["Asset"]).replace(".VN", ""),
            "Direction": r["Action"],
            "TargetZone": r["TargetZone"],
            "Conviction": r["Conviction"],
            "WhyNow": r["WhyNow"]
        })


def save_sector_heatmap_history(report_date: str, sector_heatmap_df: pd.DataFrame) -> None:
    if sector_heatmap_df.empty:
        return
    for _, r in sector_heatmap_df.iterrows():
        append_csv_row(SECTOR_HEATMAP_HISTORY_FILE, {
            "Date": report_date,
            "Sector": r["Sector"],
            "IdeaCount": r["IdeaCount"],
            "AvgScore": r["AvgScore"],
            "AvgConviction": r["AvgConviction"],
            "Heat": r["Heat"]
        })


def save_portfolio_memo_history(report_date: str, memo_text: str) -> None:
    append_csv_row(PORTFOLIO_MEMO_HISTORY_FILE, {
        "Date": report_date,
        "PortfolioMemo": memo_text[:6000]
    })


def save_action_signal_history(report_date: str, action_df: pd.DataFrame) -> None:
    if action_df.empty:
        return
    for _, r in action_df.iterrows():
        append_csv_row(ACTION_SIGNAL_HISTORY_FILE, {
            "Date": report_date,
            "Ticker": r["Ticker"],
            "Action": r["Action"],
            "Reason": r["Reason"],
            "Score": r["Score"],
            "Conviction": r["Conviction"]
        })


def save_model_portfolio_history(report_date: str, benchmark_df: pd.DataFrame) -> None:
    if benchmark_df.empty:
        return
    for _, r in benchmark_df.iterrows():
        append_csv_row(MODEL_PORTFOLIO_HISTORY_FILE, {
            "Date": report_date,
            "Bucket": r["Bucket"],
            "Ticker": r["Ticker"],
            "Weight": r["Weight"],
            "ReturnPct": r["ReturnPct"],
            "Contribution": r["Contribution"]
        })


# =========================================================
# NOTES
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


def fallback_morning_note(note_date: str, bias: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str, consensus_text: str) -> str:
    return f"""MORNING NOTE / BẢN TIN SÁNG - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Theo dõi lợi suất, lạm phát và duration.
- Monitor yields, inflation and duration.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá, thanh khoản và chính sách.
- Monitor FX, liquidity and policy direction.

3. Global Equity / Equity toàn cầu
- Tâm lý phụ thuộc risk-on/risk-off và lợi suất.
- Sentiment depends on risk-on/risk-off and yields.

4. Vietnam Equity / Equity Việt Nam
- Tập trung nhóm dẫn dắt, large caps và đồng thuận chuyên gia.
- Focus on leadership groups, large caps and expert consensus.

5. Commodity / FX
- Dầu, vàng, USD vẫn là tactical variables lớn nhất.
- Oil, gold and USD remain the biggest tactical variables.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio Positioning / CIO View
- House view: {bias}
- Giữ allocation linh hoạt và tăng risk có chọn lọc.
- Keep allocation flexible and add risk selectively.

{allocation_text}

8. Vietnam stock consensus
{consensus_text}

9. Key News
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def fallback_closing_note(note_date: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""CLOSING NOTE / BẢN TIN CUỐI NGÀY - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Đánh giá lại yields và duration view.
- Reassess yields and duration view.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá và thanh khoản.
- Monitor FX and liquidity.

3. Global Equity / Equity toàn cầu
- Xác định mức cải thiện hay xấu đi của risk sentiment.
- Determine whether risk sentiment improved or deteriorated.

4. Vietnam Equity / Equity Việt Nam
- Kiểm tra độ rộng, dòng tiền và nhóm dẫn dắt.
- Check breadth, flows and leadership groups.

5. Commodity / FX
- Theo dõi dầu, vàng và USD.
- Monitor oil, gold and USD.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Changes vs Morning / Portfolio Implications
{allocation_text}
"""


def fallback_ic_note(note_date: str, news_df: pd.DataFrame, expert_notes: str, allocation_text: str, trade_ideas_text: str) -> str:
    return f"""IC NOTE / GHI CHÚ IC - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Duration nên giữ trung lập nếu yields chưa dịu rõ.
- Duration should remain neutral if yields have not clearly eased.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi FX và thanh khoản nội địa.
- Monitor FX and domestic liquidity.

3. Global Equity / Equity toàn cầu
- Positioning nên linh hoạt theo regime.
- Positioning should stay flexible by regime.

4. Vietnam Equity / Equity Việt Nam
- Ưu tiên các mã có consensus mạnh và large caps dẫn dắt.
- Prefer names with strong consensus and leadership large caps.

5. Commodity / FX
- Dầu, vàng, USD là các công cụ tactical và hedge.
- Oil, gold and USD are tactical and hedging tools.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio implication / Hàm ý danh mục
{allocation_text}

8. Trade ideas / Ý tưởng đầu tư
{trade_ideas_text[:1800]}
"""


def generate_morning_note(api_key: str, model: str, note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str, allocation_text: str, consensus_text: str) -> str:
    system_prompt = """You are a senior CIO strategist.
Write a STRICTLY BILINGUAL Vietnamese-English Morning Note.
Focus on interpretation, cross-asset implications, portfolio action, risk, catalyst, and allocation.
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

VIETNAM CONSENSUS:
{consensus_text}

USER NOTES:
{normalize_text(user_news)[:700]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1900)
    return out if out else fallback_morning_note(note_date, bias, news_df, expert_notes, allocation_text, consensus_text)


def generate_closing_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are an end-of-day strategist. Write a STRICTLY BILINGUAL Vietnamese-English Closing Note."""
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


def generate_ic_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str, trade_ideas_text: str) -> str:
    system_prompt = """You are a PM strategist writing for CIO and Investment Committee. Write a STRICTLY BILINGUAL Vietnamese-English IC Note."""
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
TRADE IDEAS:
{trade_ideas_text[:2500]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 2000)
    return out if out else fallback_ic_note(note_date, news_df, expert_notes, allocation_text, trade_ideas_text)


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
        chunks = [raw_line[i:i+110] for i in range(0, len(raw_line), 110)] or [""]
        for line in chunks:
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
    append_csv_row(MARKET_HISTORY_FILE, {
        "Date": note_date,
        "MorningNote": morning_note[:6000],
        "ClosingNote": closing_note[:6000]
    })


def save_ic_history(note_date: str, ic_note: str) -> None:
    append_csv_row(IC_HISTORY_FILE, {"Date": note_date, "ICNote": ic_note[:6000]})


def save_trade_ideas_history(note_date: str, trade_ideas: str) -> None:
    append_csv_row(TRADE_IDEAS_HISTORY_FILE, {"Date": note_date, "TradeIdeas": trade_ideas[:6000]})


def build_email_bundle(api_key: str, model: str, report_date: str, run_label: str, morning_note: str, closing_note: str, ic_note: str, trade_ideas: str, allocation_text: str, portfolio_memo: str) -> Tuple[str, str]:
    core_note = f"{morning_note}\n\n{closing_note}\n\n{portfolio_memo}"
    subject, body = generate_email_body(api_key, model, core_note + "\n\n" + allocation_text, report_date, run_label)
    top_idea = extract_top_idea(trade_ideas)
    if portfolio_memo:
        body += "\n\n=== Portfolio Memo ===\n" + portfolio_memo
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

    consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
    consensus_text = build_vn_consensus_text(consensus_df)

    allocation = compute_portfolio_allocation_v65(market_df, news_df_ai, consensus_df)
    allocation_text = format_allocation_v65(allocation)

    trade_ideas_df = generate_ranked_trade_ideas_v65(
        market_df,
        news_df_ai,
        consensus_df=consensus_df,
        ideas_count=int(cfg.get("trade_ideas_count", 8))
    )
    trade_ideas_text = format_trade_ideas_v65(trade_ideas_df, consensus_text)

    benchmark_df = compute_model_portfolio_benchmark(MODEL_PORTFOLIO_V65, market_df)
    action_df = generate_action_signals(trade_ideas_df, MODEL_PORTFOLIO_V65)
    sector_heatmap_df = build_sector_conviction_heatmap(trade_ideas_df, VN_SECTOR_MAP)
    portfolio_memo = build_portfolio_memo(allocation_text, trade_ideas_df, action_df, benchmark_df)

    morning_note = generate_morning_note(
        api_key, model, report_date, cfg.get("house_view", "Neutral"),
        market_df, news_df_ai, user_news, expert_notes, allocation_text, consensus_text
    )
    closing_note = generate_closing_note(
        api_key, model, report_date, market_df, news_df_ai, expert_notes, allocation_text
    )
    ic_note = generate_ic_note(
        api_key, model, report_date, market_df, news_df_ai, expert_notes, allocation_text, trade_ideas_text
    )

    email_subject, email_body = build_email_bundle(
        api_key, model, report_date, "MANUAL",
        morning_note, closing_note, ic_note, trade_ideas_text, allocation_text, portfolio_memo
    )

    save_target_price_history(report_date, trade_ideas_df)
    save_sector_heatmap_history(report_date, sector_heatmap_df)
    save_portfolio_memo_history(report_date, portfolio_memo)
    save_action_signal_history(report_date, action_df)
    save_model_portfolio_history(report_date, benchmark_df)

    return {
        "market_df": market_df,
        "news_df": news_df,
        "news_df_ai": news_df_ai,
        "consensus_df": consensus_df,
        "consensus_text": consensus_text,
        "allocation_text": allocation_text,
        "trade_ideas_df": trade_ideas_df,
        "benchmark_df": benchmark_df,
        "action_df": action_df,
        "sector_heatmap_df": sector_heatmap_df,
        "portfolio_memo": portfolio_memo,
        "morning_note": morning_note,
        "closing_note": closing_note,
        "ic_note": ic_note,
        "trade_ideas": trade_ideas_text,
        "email_subject": email_subject,
        "email_body": email_body,
    }


# =========================================================
# SCHEDULER
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
    subject, body = build_email_bundle(
        api_key, model, report_date, "09:00 AUTO",
        result["morning_note"], result["closing_note"], result["ic_note"],
        result["trade_ideas"], result["allocation_text"], result["portfolio_memo"]
    )
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
# UI HELPERS
# =========================================================
def init_session():
    defaults = {
        "current_user": None,
        "market_df": pd.DataFrame(),
        "news_df": pd.DataFrame(),
        "news_df_ai": pd.DataFrame(),
        "trade_ideas_df": pd.DataFrame(),
        "consensus_df": pd.DataFrame(),
        "benchmark_df": pd.DataFrame(),
        "action_df": pd.DataFrame(),
        "sector_heatmap_df": pd.DataFrame(),
        "consensus_text": "",
        "allocation_text": "",
        "portfolio_memo": "",
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


def login_screen():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
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


# =========================================================
# ADMIN TABS
# =========================================================
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

    if users:
        st.markdown("#### Cập nhật user")
        selected_username = st.selectbox("Chọn user", [u["username"] for u in users], key="edit_user_select")
        selected = next((u for u in users if u["username"] == selected_username), None)
        if selected:
            edit_role = st.selectbox("Role mới", ["admin", "analyst", "viewer"], index=["admin", "analyst", "viewer"].index(selected.get("role", "viewer")), key="edit_role")
            edit_active = st.checkbox("Active", value=selected.get("active", True), key="edit_active")
            reset_password = st.text_input("Reset password", type="password", key="reset_password")
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
    st.dataframe(pd.DataFrame([{"Group": k, "Emails": ", ".join(v)} for k, v in groups.items()]), use_container_width=True)

    new_group_name = st.text_input("Tên nhóm mới", key="new_group_name")
    new_group_emails = st.text_area("Emails", key="new_group_emails")
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


# =========================================================
# MAIN APP
# =========================================================
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

    top_left, top_right = st.columns([8, 1])
    with top_right:
        if st.button("Logout"):
            log_audit(current_user.get("username", ""), "logout")
            st.session_state["current_user"] = None
            st.rerun()

    st.sidebar.title("⚙️ Cấu hình")
    model = st.sidebar.text_input("Model", value=cfg.get("model", DEFAULT_MODEL))
    report_date = str(st.sidebar.date_input("Ngày báo cáo"))
    house_view = st.sidebar.selectbox(
        "House View",
        ["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"],
        index=["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"].index(cfg.get("house_view", "Neutral"))
    )
    smart_mode = st.sidebar.checkbox("Smart mode", value=cfg.get("smart_mode", True))
    recipients_manual = st.sidebar.text_area("Recipients thủ công", value=recipients_default, height=80)
    trade_ideas_count = st.sidebar.number_input("Trade ideas count", min_value=3, max_value=12, value=int(cfg.get("trade_ideas_count", 8)))
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
                cfg["trade_ideas_count"] = int(trade_ideas_count)
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

    tab_names = [
        "Dashboard",
        "Daily Notes",
        "Allocation Deep Dive",
        "Trade Ideas",
        "Portfolio Benchmark",
        "Conviction Heatmap",
        "Portfolio Memo",
        "Action Signals",
        "Email"
    ]
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
                    cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in vn_show.columns]
                    st.write(vn_show[cols].to_html(escape=False, index=False), unsafe_allow_html=True)
            with b:
                st.markdown("#### Global News")
                if not global_df.empty:
                    global_show = global_df.copy()
                    global_show["ArticleLink"] = global_show["Link"].apply(lambda x: f'<a href="{x}" target="_blank">Open article</a>' if pd.notna(x) and str(x).strip() else "")
                    cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in global_show.columns]
                    st.write(global_show[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

            st.text_area("Signals", build_top_actionable_signals(st.session_state["news_df_ai"], 5), height=120)

    with tab_map["Daily Notes"]:
        if has_permission("generate_notes"):
            a, b, c = st.columns(3)
            with a:
                if st.button("Generate Morning Note", use_container_width=True):
                    if not st.session_state["allocation_text"]:
                        consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
                        consensus_text = build_vn_consensus_text(consensus_df)
                        allocation = compute_portfolio_allocation_v65(st.session_state["market_df"], st.session_state["news_df_ai"], consensus_df)
                        st.session_state["consensus_df"] = consensus_df
                        st.session_state["consensus_text"] = consensus_text
                        st.session_state["allocation_text"] = format_allocation_v65(allocation)
                    st.session_state["morning_note"] = generate_morning_note(
                        api_key, model, report_date, house_view, st.session_state["market_df"], st.session_state["news_df_ai"],
                        st.session_state["user_news"], st.session_state["expert_notes"], st.session_state["allocation_text"], st.session_state["consensus_text"]
                    )
            with b:
                if st.button("Generate Closing Note", use_container_width=True):
                    if not st.session_state["allocation_text"]:
                        consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
                        allocation = compute_portfolio_allocation_v65(st.session_state["market_df"], st.session_state["news_df_ai"], consensus_df)
                        st.session_state["allocation_text"] = format_allocation_v65(allocation)
                    st.session_state["closing_note"] = generate_closing_note(
                        api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"],
                        st.session_state["expert_notes"], st.session_state["allocation_text"]
                    )
            with c:
                if st.button("Generate IC Note", use_container_width=True):
                    if not st.session_state["trade_ideas"]:
                        consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
                        trade_df = generate_ranked_trade_ideas_v65(st.session_state["market_df"], st.session_state["news_df_ai"], consensus_df, int(trade_ideas_count))
                        st.session_state["trade_ideas_df"] = trade_df
                        st.session_state["trade_ideas"] = format_trade_ideas_v65(trade_df, build_vn_consensus_text(consensus_df))
                    if not st.session_state["allocation_text"]:
                        consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
                        allocation = compute_portfolio_allocation_v65(st.session_state["market_df"], st.session_state["news_df_ai"], consensus_df)
                        st.session_state["allocation_text"] = format_allocation_v65(allocation)
                    st.session_state["ic_note"] = generate_ic_note(
                        api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"],
                        st.session_state["expert_notes"], st.session_state["allocation_text"], st.session_state.get("trade_ideas", "")
                    )

        st.markdown("### Morning Note / Bản tin sáng")
        if st.session_state["morning_note"]:
            st.text_area("Morning Note", st.session_state["morning_note"], height=320)
            if has_permission("export_notes"):
                render_export_buttons("Morning Note", st.session_state["morning_note"], f"Morning_Note_{report_date}")

        st.markdown("### Closing Note / Bản tin cuối ngày")
        if st.session_state["closing_note"]:
            st.text_area("Closing Note", st.session_state["closing_note"], height=320)
            if has_permission("export_notes"):
                render_export_buttons("Closing Note", st.session_state["closing_note"], f"Closing_Note_{report_date}")

        st.markdown("### IC Note / Ghi chú IC")
        if st.session_state["ic_note"]:
            st.text_area("IC Note", st.session_state["ic_note"], height=360)
            if has_permission("export_notes"):
                render_export_buttons("IC Note", st.session_state["ic_note"], f"IC_Note_{report_date}")

    with tab_map["Allocation Deep Dive"]:
        if st.button("Compute Allocation v6.5", use_container_width=True):
            consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
            st.session_state["consensus_df"] = consensus_df
            st.session_state["consensus_text"] = build_vn_consensus_text(consensus_df)
            allocation = compute_portfolio_allocation_v65(st.session_state["market_df"], st.session_state["news_df_ai"], consensus_df)
            st.session_state["allocation_text"] = format_allocation_v65(allocation)

        if st.session_state["allocation_text"]:
            st.text_area("Allocation Output", st.session_state["allocation_text"], height=500)

    with tab_map["Trade Ideas"]:
        if st.button("Generate Ranked Trade Ideas v6.5", use_container_width=True):
            consensus_df = summarize_vn_expert_recommendations(VN_EXPERT_RECOMMENDATIONS)
            st.session_state["consensus_df"] = consensus_df
            st.session_state["consensus_text"] = build_vn_consensus_text(consensus_df)
            st.session_state["trade_ideas_df"] = generate_ranked_trade_ideas_v65(
                st.session_state["market_df"],
                st.session_state["news_df_ai"],
                consensus_df=consensus_df,
                ideas_count=int(trade_ideas_count)
            )
            st.session_state["trade_ideas"] = format_trade_ideas_v65(st.session_state["trade_ideas_df"], st.session_state["consensus_text"])

        if isinstance(st.session_state.get("trade_ideas_df"), pd.DataFrame) and not st.session_state["trade_ideas_df"].empty:
            st.dataframe(st.session_state["trade_ideas_df"], use_container_width=True)

        if isinstance(st.session_state.get("consensus_df"), pd.DataFrame) and not st.session_state["consensus_df"].empty:
            st.markdown("### Vietnam Expert Consensus Stocks")
            st.dataframe(st.session_state["consensus_df"], use_container_width=True)

        if st.session_state["trade_ideas"]:
            st.text_area("Trade Ideas", st.session_state["trade_ideas"], height=420)
            st.text_area("Top Idea", extract_top_idea(st.session_state["trade_ideas"]), height=150)

    with tab_map["Portfolio Benchmark"]:
        if st.button("Compute Portfolio Benchmark", use_container_width=True):
            st.session_state["benchmark_df"] = compute_model_portfolio_benchmark(MODEL_PORTFOLIO_V65, st.session_state["market_df"])

        if isinstance(st.session_state.get("benchmark_df"), pd.DataFrame) and not st.session_state["benchmark_df"].empty:
            bdf = st.session_state["benchmark_df"]
            st.dataframe(bdf, use_container_width=True)
            st.write(f"**Model portfolio return:** {bdf.attrs.get('portfolio_return', 0.0)}%")
            st.write(f"**VNINDEX return:** {bdf.attrs.get('vnindex_return', 0.0)}%")
            st.write(f"**Excess return:** {bdf.attrs.get('excess_return', 0.0)}%")

    with tab_map["Conviction Heatmap"]:
        if st.button("Build Sector Heatmap", use_container_width=True):
            if isinstance(st.session_state.get("trade_ideas_df"), pd.DataFrame) and not st.session_state["trade_ideas_df"].empty:
                st.session_state["sector_heatmap_df"] = build_sector_conviction_heatmap(st.session_state["trade_ideas_df"], VN_SECTOR_MAP)

        if isinstance(st.session_state.get("sector_heatmap_df"), pd.DataFrame) and not st.session_state["sector_heatmap_df"].empty:
            st.dataframe(st.session_state["sector_heatmap_df"], use_container_width=True)

    with tab_map["Portfolio Memo"]:
        if st.button("Build Portfolio Memo", use_container_width=True):
            if isinstance(st.session_state.get("benchmark_df"), pd.DataFrame) and not st.session_state["benchmark_df"].empty:
                bench = st.session_state["benchmark_df"]
            else:
                bench = compute_model_portfolio_benchmark(MODEL_PORTFOLIO_V65, st.session_state["market_df"])
                st.session_state["benchmark_df"] = bench

            if isinstance(st.session_state.get("action_df"), pd.DataFrame) and not st.session_state["action_df"].empty:
                actions = st.session_state["action_df"]
            else:
                actions = generate_action_signals(st.session_state.get("trade_ideas_df", pd.DataFrame()), MODEL_PORTFOLIO_V65)
                st.session_state["action_df"] = actions

            st.session_state["portfolio_memo"] = build_portfolio_memo(
                st.session_state.get("allocation_text", ""),
                st.session_state.get("trade_ideas_df", pd.DataFrame()),
                actions,
                bench
            )

        if st.session_state["portfolio_memo"]:
            st.text_area("Portfolio Memo", st.session_state["portfolio_memo"], height=260)
            if has_permission("export_notes"):
                render_export_buttons("Portfolio Memo", st.session_state["portfolio_memo"], f"Portfolio_Memo_{report_date}")

    with tab_map["Action Signals"]:
        if st.button("Generate Action Signals", use_container_width=True):
            if isinstance(st.session_state.get("trade_ideas_df"), pd.DataFrame) and not st.session_state["trade_ideas_df"].empty:
                st.session_state["action_df"] = generate_action_signals(st.session_state["trade_ideas_df"], MODEL_PORTFOLIO_V65)

        if isinstance(st.session_state.get("action_df"), pd.DataFrame) and not st.session_state["action_df"].empty:
            st.dataframe(st.session_state["action_df"], use_container_width=True)

    with tab_map["Email"]:
        groups = load_recipient_groups()
        selected_groups = st.multiselect("Chọn nhóm người nhận", list(groups.keys()), default=[])
        base_note = st.session_state["morning_note"] or st.session_state["closing_note"] or ""

        if st.button("Generate Email", use_container_width=True) and base_note:
            if not st.session_state["portfolio_memo"] and isinstance(st.session_state.get("benchmark_df"), pd.DataFrame):
                st.session_state["portfolio_memo"] = build_portfolio_memo(
                    st.session_state.get("allocation_text", ""),
                    st.session_state.get("trade_ideas_df", pd.DataFrame()),
                    st.session_state.get("action_df", pd.DataFrame()),
                    st.session_state.get("benchmark_df", pd.DataFrame())
                )

            subject, body = build_email_bundle(
                api_key, model, report_date, "MANUAL",
                st.session_state.get("morning_note", ""),
                st.session_state.get("closing_note", ""),
                st.session_state.get("ic_note", ""),
                st.session_state.get("trade_ideas", ""),
                st.session_state.get("allocation_text", ""),
                st.session_state.get("portfolio_memo", "")
            )
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
                    else:
                        st.error(msg)
            with c2:
                if st.button("Send to Selected Groups", use_container_width=True):
                    recipients_list = get_emails_from_groups(selected_groups)
                    ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
                    log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)
                    if ok:
                        st.success("Đã gửi email theo nhóm.")
                    else:
                        st.error(msg)

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
