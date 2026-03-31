from __future__ import annotations

import hashlib
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


APP_TITLE = "📊 Analyst Dashboard - TVAM- Nhung"
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
    "VN Equity - Themes": ["VCI.VN", "CTG.VN", "BID.VN", "KDH.VN", "NLG.VN", "PVD.VN", "DGC.VN"],
    "VN Broker Favorites": ["FPT.VN", "MWG.VN", "ACB.VN", "TCB.VN", "VCB.VN", "SSI.VN", "HPG.VN", "PNJ.VN"],
    "VN Macro Products": ["VNINDEX", "USD/VND", "Lãi suất VN", "Trái phiếu Chính phủ VN", "Vàng", "Dầu"],
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


# =========================================================
# FILE / CONFIG HELPERS
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


def ensure_files() -> None:
    if not CONFIG_FILE.exists():
        save_json(CONFIG_FILE, {
            "model": DEFAULT_MODEL,
            "house_view": "Neutral",
            "default_sender_email": "",
            "default_recipients": "",
            "smart_mode": True,
            "auto_send_after_run_all": False,
            "trade_ideas_count": 7,
            "auto_user_news": "",
            "auto_expert_notes": ""
        })

    if not CACHE_FILE.exists():
        save_json(CACHE_FILE, {})

    init_csv(EMAIL_LOG_FILE, ["SendTime", "Sender", "Recipients", "Subject", "Status", "Error"])
    init_csv(MARKET_HISTORY_FILE, ["Date", "MorningNote", "ClosingNote"])
    init_csv(IC_HISTORY_FILE, ["Date", "ICNote"])
    init_csv(TRADE_IDEAS_HISTORY_FILE, ["Date", "TradeIdeas"])


def load_config() -> dict:
    return load_json(CONFIG_FILE, {})


def save_config(cfg: dict) -> None:
    save_json(CONFIG_FILE, cfg)


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


def load_cache() -> dict:
    return load_json(CACHE_FILE, {})


def save_cache(cache: dict) -> None:
    save_json(CACHE_FILE, cache)


def ai_is_available(api_key: str) -> bool:
    return bool(api_key) and (OpenAI is not None)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def get_runtime_value(config_value: str, secret_name: str) -> str:
    secret_val = get_secret(secret_name, "")
    return secret_val if secret_val else (config_value or "")


# =========================================================
# AI HELPERS
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

    vn_source_keywords = ["vnexpress", "vietstock", "cafef", "ndh", "stockbiz", "thoibaotaichinhvietnam"]
    vn_content_keywords = [
        "vietnam", "viet nam", "vnindex", "hose", "hnx", "upcom", "sbv",
        "tỷ giá", "ty gia", "lãi suất", "trái phiếu", "chứng khoán", "doanh nghiệp",
        "ngân hàng", "vnd", "fpt", "vcb", "hpg", "mbb", "ssi", "vhm", "gas"
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
    for k in ["vnindex", "vietnam", "sbv", "usd", "oil", "bank", "lãi suất", "tỷ giá", "ngân hàng", "fpt", "vcb", "hpg"]:
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
    return "\n".join(
        f"- {r['Asset']}: {r['Price']} ({r['ChangePct']}%)"
        for _, r in market_df.iterrows()
    )


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
    return "\n".join(
        f"- [{r['AssetClass']}] {r['Title']} | VN Impact {r['VNImpact']}"
        for _, r in df.iterrows()
    )


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
# EXPERT / FUND SUMMARY
# =========================================================
def build_expert_fund_summary(user_expert_notes: str = "") -> str:
    base = """
- SSI: Ưu tiên theo dõi large caps, câu chuyện nâng hạng và dòng vốn quay lại / Focus on large caps, upgrade story and returning flows.
- VNDirect: Theo dõi biến động lãi suất, tỷ giá, nhóm dẫn dắt trong nước / Monitor rates, FX and domestic leadership.
- Funds: Ưu tiên allocation linh hoạt, quản trị risk chặt chẽ / Prefer flexible allocation and tight risk management.
"""
    if normalize_text(user_expert_notes):
        base += "\n- Additional expert views / Ghi chú thêm:\n" + normalize_text(user_expert_notes)[:1200]
    return base


# =========================================================
# V6.1 ALLOCATION ENGINE
# =========================================================
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

    equity_risk_score = 0

    if oil_chg > 1:
        allocation["Commodity"]["Oil"] = "Overweight"
        allocation["FixedIncome"]["Global"] = "Underweight"
        equity_risk_score -= 1

    if gold_chg > 0.5:
        allocation["Commodity"]["Gold"] = "Overweight"

    if usd_chg > 0.5:
        allocation["FX"]["USD"] = "Long"
        allocation["Equity"]["Global"] = "Underweight"
        equity_risk_score -= 1

    if us10y_chg > 0.5:
        allocation["FixedIncome"]["Global"] = "Underweight"
    elif us10y_chg < -0.3:
        allocation["FixedIncome"]["Global"] = "Overweight"

    if vn_chg > 1:
        allocation["Equity"]["Vietnam"] = "Overweight"
        equity_risk_score += 1
    elif vn_chg < -1:
        allocation["Equity"]["Vietnam"] = "Underweight"

    if spx_chg > 0.5:
        allocation["Equity"]["Global"] = "Overweight"
        equity_risk_score += 1
    elif spx_chg < -0.8:
        allocation["Equity"]["Global"] = "Underweight"

    if usd_chg > 0.7:
        allocation["FixedIncome"]["Vietnam"] = "Neutral"
    elif usd_chg < 0 and vn_chg > 0.5:
        allocation["FixedIncome"]["Vietnam"] = "Neutral"

    if equity_risk_score >= 1 and usd_chg <= 0.5:
        allocation["Regime"] = "Risk-on"
    elif equity_risk_score <= -1 or usd_chg > 0.5 or oil_chg > 1.5:
        allocation["Regime"] = "Risk-off"
    else:
        allocation["Regime"] = "Neutral / Transition"

    return allocation


def format_allocation(allocation: dict) -> str:
    lines = [
        "PORTFOLIO ALLOCATION / PHÂN BỔ DANH MỤC",
        f"- Market Regime / Chế độ thị trường: {allocation.get('Regime', 'Neutral')}",
        "",
        "EQUITY:"
    ]
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


# =========================================================
# V6.1 TRADE IDEA SCORING
# =========================================================
def score_trade_idea(asset: str, market_df: pd.DataFrame, news_df: pd.DataFrame) -> dict:
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
            rationale.append("Oil momentum positive")
        elif oil_chg < -1:
            score -= 1

    if "gold" in asset_lower:
        if gold_chg > 0.5 or usd_chg > 0.4:
            score += 1
            rationale.append("Defensive demand supports gold")

    if "usd" in asset_lower:
        if usd_chg > 0.5:
            score += 2
            rationale.append("USD strength confirmed")
        elif usd_chg < -0.5:
            score -= 1

    if "vnindex" in asset_lower or ".vn" in asset_lower or "vietnam" in asset_lower:
        if vn_chg > 0.7:
            score += 1.5
            rationale.append("Vietnam equity momentum supportive")
        elif vn_chg < -0.8:
            score -= 1

    if "global equity" in asset_lower:
        if spx_chg > 0.5:
            score += 1.5
        elif spx_chg < -0.8:
            score -= 1

    if "bank" in asset_lower and vn_chg > 0:
        score += 1
        rationale.append("Domestic leadership supportive")

    if not news_df.empty:
        matches = news_df[
            news_df["Title"].str.contains(asset.split()[0], case=False, na=False) |
            news_df["Summary"].str.contains(asset.split()[0], case=False, na=False)
        ]
        score += min(len(matches) * 0.5, 2.0)
        if len(matches) > 0:
            rationale.append(f"News flow support: {len(matches)} items")

    score = max(1.0, min(score, 10.0))

    if score >= 8:
        confidence = "High"
    elif score >= 6:
        confidence = "Medium"
    else:
        confidence = "Low"

    direction = "Watch"
    if score >= 8:
        direction = "Overweight / Tăng tỷ trọng"
    elif score >= 6.5:
        direction = "Positive Watch / Theo dõi tích cực"
    elif score <= 4.5:
        direction = "Avoid / Hạn chế"

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
    universe = [
        "Oil",
        "Gold",
        "USD",
        "VNINDEX",
        "Vietnam Banks",
        "FPT.VN",
        "HPG.VN",
        "VCB.VN",
        "SSI.VN",
        "Global Equity"
    ]
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
            f"- Direction / Hướng hành động: {row['Direction']}\n"
            f"- Score: {row['Score']}\n"
            f"- Confidence: {row['Confidence']}\n"
            f"- Horizon / Khung thời gian: {row['Horizon']}\n"
            f"- Rationale / Luận điểm: {row['Rationale']}"
        )
    return "\n".join(lines)


def extract_top_idea(trade_ideas_text: str) -> str:
    text = normalize_text(trade_ideas_text)
    if not text:
        return "Chưa có ý tưởng nổi bật / No top idea available."
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        return blocks[1]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return "\n".join(lines[:8]) if lines else "Chưa có ý tưởng nổi bật / No top idea available."


# =========================================================
# NOTE GENERATION
# =========================================================
def fallback_morning_note(note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str, allocation_text: str) -> str:
    return f"""MORNING NOTE / BẢN TIN SÁNG - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Theo dõi lợi suất, lạm phát và định vị duration vì dầu và USD đang ảnh hưởng rõ tới thị trường trái phiếu.
- Monitor yields, inflation and duration positioning as oil and USD are materially affecting bond markets.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá, thanh khoản và định hướng điều hành để đánh giá môi trường lãi suất nội địa.
- Monitor FX, liquidity and policy direction to assess the domestic rate environment.

3. Global Equity / Equity toàn cầu
- Tâm lý cổ phiếu toàn cầu đang phụ thuộc vào risk-on/risk-off, dầu và lợi suất.
- Global equity sentiment is being driven by risk-on/risk-off dynamics, oil and yields.

4. Vietnam Equity / Equity Việt Nam
- Tập trung vào nhóm dẫn dắt, thanh khoản và dòng vốn ngoại để xác nhận xu hướng.
- Focus on leadership sectors, liquidity and foreign flows to confirm trend strength.

5. Commodity / FX
- Dầu, vàng, USD và USD/VND vẫn là biến số chiến thuật lớn nhất cho danh mục.
- Oil, gold, USD and USD/VND remain the largest tactical variables for portfolios.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio Positioning & CIO View / Phân bổ danh mục & góc nhìn CIO
- House view: {bias}
- Ưu tiên allocation linh hoạt và chỉ tăng rủi ro khi có xác nhận rõ hơn.
- Keep allocation flexible and only add risk with clearer confirmation.

{allocation_text}

8. Additional notes / Ghi chú thêm
{user_news[:500] if user_news else "- Không có. / None."}

9. Key news / Tin nổi bật
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def fallback_closing_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""CLOSING NOTE / BẢN TIN CUỐI NGÀY - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Cần đánh giá lại biến động lợi suất và hàm ý cho duration sau phiên giao dịch gần nhất.
- Reassess yield moves and their implication for duration after the latest session.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá, thanh khoản và tín hiệu điều hành để cập nhật view nội địa.
- Monitor FX, liquidity and policy signals to update the domestic view.

3. Global Equity / Equity toàn cầu
- Xác định xem risk sentiment đang cải thiện hay xấu đi so với đầu ngày.
- Determine whether risk sentiment improved or deteriorated versus the morning setup.

4. Vietnam Equity / Equity Việt Nam
- Kiểm tra độ rộng, thanh khoản và nhóm dẫn dắt để đánh giá sức bền của nhịp tăng/giảm.
- Check breadth, liquidity and leadership to assess the durability of the move.

5. Commodity / FX
- Dầu, vàng và USD tiếp tục là biến số chi phối tactical positioning.
- Oil, gold and USD continue to drive tactical positioning.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Changes vs Morning / What Changed and Next Session Plan
- Giữ kỷ luật position sizing và cập nhật phân bổ theo tín hiệu mới.
- Maintain disciplined position sizing and update allocation based on new signals.

{allocation_text}

8. Key news / Tin nổi bật
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def fallback_ic_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    return f"""IC NOTE / GHI CHÚ IC - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Duration nên giữ trung lập cho tới khi biến động dầu và lợi suất dịu lại.
- Duration should stay neutral until oil and yield volatility eases.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Ưu tiên theo dõi tỷ giá, thanh khoản và đường cong lợi suất nội địa.
- Focus on FX, liquidity and the domestic yield curve.

3. Global Equity / Equity toàn cầu
- Positioning nên linh hoạt theo mức độ risk-on/risk-off thay vì neo vào 1 kịch bản duy nhất.
- Positioning should remain flexible based on risk-on/risk-off conditions rather than one fixed scenario.

4. Vietnam Equity / Equity Việt Nam
- Ưu tiên large caps, nhóm dẫn dắt và câu chuyện nâng hạng nếu dòng tiền ủng hộ.
- Prefer large caps, leadership groups and the upgrade story if flows remain supportive.

5. Commodity / FX
- Dầu, vàng, USD và USD/VND tác động trực tiếp đến tactical allocation và hedge.
- Oil, gold, USD and USD/VND directly affect tactical allocation and hedging.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio implication / Hàm ý danh mục
- Overweight có chọn lọc ở equity Việt Nam, neutral duration, theo dõi commodity shock và USD strength.
- Selective overweight in Vietnam equities, neutral duration, and monitor commodity shocks and USD strength.

{allocation_text}

8. Key news / Tin nổi bật
{build_vn_global_news_brief(news_df, 4, 4)}
"""


def generate_morning_note(api_key: str, model: str, note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are a senior CIO strategist at a global asset management firm.

Your audience includes CIO, Portfolio Managers, and Investment Committee.

Your job is NOT to summarize news.
Your job is to interpret markets, identify cross-asset implications, and provide actionable portfolio positioning.

Write a STRICTLY BILINGUAL Vietnamese-English Morning Note.

Mandatory structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. Portfolio Positioning & CIO View
8. Risks & Catalysts

Requirements:
- Every heading must be bilingual.
- Every bullet must include BOTH Vietnamese and English.
- Explain WHY it matters and SO WHAT for portfolios.
- Explicitly mention duration view, equity positioning, FX/commodity impact.
- Use institutional tone, concise, high signal, no fluff.
- Include overweight / neutral / underweight where relevant.
"""
    user_prompt = f"""
Date: {note_date}
House View: {bias}

MARKET SNAPSHOT:
{build_market_highlights(market_df)}

KEY NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

TOP SIGNALS:
{build_top_actionable_signals(news_df, 5)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}

CURRENT ALLOCATION ENGINE OUTPUT:
{allocation_text}

USER NOTES:
{normalize_text(user_news)[:700]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1900)
    return out if out else fallback_morning_note(note_date, bias, market_df, news_df, user_news, expert_notes, allocation_text)


def generate_closing_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are an end-of-day strategist supporting CIO / IC / Portfolio Managers.

Write a STRICTLY BILINGUAL Vietnamese-English Closing Note.

Mandatory structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. Changes vs Morning / Portfolio Implications
8. Risks & What to Watch Next

Requirements:
- Every heading must be bilingual.
- Every bullet must contain both Vietnamese and English.
- Focus on interpretation and portfolio action, not headlines.
- State what changed, why it matters, and what should be done next.
"""
    user_prompt = f"""
Date: {note_date}

MARKET SNAPSHOT:
{build_market_highlights(market_df)}

KEY NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

TOP SIGNALS:
{build_top_actionable_signals(news_df, 5)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}

CURRENT ALLOCATION ENGINE OUTPUT:
{allocation_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1900)
    return out if out else fallback_closing_note(note_date, market_df, news_df, expert_notes, allocation_text)


def generate_ic_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str, allocation_text: str) -> str:
    system_prompt = """You are a PM strategist writing for Investment Committee / Portfolio Managers.

Write a STRICTLY BILINGUAL Vietnamese-English IC / PM Note.

Mandatory structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. Portfolio Implication / Suggested IC Actions
8. Risks, Catalysts, and What Could Change the View

Requirements:
- Every heading must be bilingual.
- Every bullet must contain both Vietnamese and English.
- Think like a fund manager making real allocation decisions.
- Explicitly state overweight / neutral / underweight when appropriate.
- Distinguish tactical (1-4 weeks) vs strategic (3-6 months) if useful.
"""
    user_prompt = f"""
Date: {note_date}

MARKET SNAPSHOT:
{build_market_highlights(market_df)}

KEY NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

TOP SIGNALS:
{build_top_actionable_signals(news_df, 5)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}

CURRENT ALLOCATION ENGINE OUTPUT:
{allocation_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 2000)
    return out if out else fallback_ic_note(note_date, market_df, news_df, expert_notes, allocation_text)


# =========================================================
# EMAIL
# =========================================================
def generate_email_body(api_key: str, model: str, core_note: str, report_date: str, run_label: str) -> Tuple[str, str]:
    subject = f"[{run_label}] Cập nhật thị trường | Market Update - {report_date}"
    system_prompt = """Write a STRICTLY BILINGUAL Vietnamese-English market update email.

Required sections:
1. Market Summary
2. Key Drivers
3. Portfolio Positioning
4. Top Trade Idea
5. What to Watch

Rules:
- Greeting must be bilingual.
- Every major paragraph must contain both Vietnamese and English.
- Keep it concise and institutional.
- End with:
Trân trọng.
Best regards.
"""
    out = cached_ai_call(api_key, model, system_prompt, core_note[:3500], 1200)
    if out:
        return subject, out

    body = f"""Kính gửi anh/chị,
Dear all,

Dưới đây là bản cập nhật thị trường ngày {report_date}.
Please find below the market update for {report_date}.

{core_note}

Trân trọng.
Best regards.
"""
    return subject, body


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
        "ClosingNote": closing_note[:6000],
    })


def save_ic_history(note_date: str, ic_note: str) -> None:
    append_csv_row(IC_HISTORY_FILE, {"Date": note_date, "ICNote": ic_note[:6000]})


def save_trade_ideas_history(note_date: str, trade_ideas: str) -> None:
    append_csv_row(TRADE_IDEAS_HISTORY_FILE, {"Date": note_date, "TradeIdeas": trade_ideas[:6000]})


def build_email_bundle(api_key: str, model: str, report_date: str, run_label: str, core_note: str, ic_note: str, trade_ideas: str, allocation_text: str) -> Tuple[str, str]:
    subject, body = generate_email_body(api_key, model, core_note + "\n\n" + allocation_text, report_date, run_label)

    top_idea = extract_top_idea(trade_ideas)
    if allocation_text:
        body += "\n\n=== Portfolio Allocation | Phân bổ danh mục ===\n" + allocation_text
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

    morning_note = generate_morning_note(
        api_key, model, report_date, cfg.get("house_view", "Neutral"),
        market_df, news_df_ai, user_news, expert_notes, allocation_text
    )

    closing_note = generate_closing_note(
        api_key, model, report_date,
        market_df, news_df_ai, expert_notes, allocation_text
    )

    ic_note = generate_ic_note(
        api_key, model, report_date,
        market_df, news_df_ai, expert_notes, allocation_text
    )

    email_subject, email_body = build_email_bundle(
        api_key, model, report_date, "MANUAL",
        morning_note, ic_note, trade_ideas_text, allocation_text
    )

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


def run_auto_job() -> None:
    ensure_files()
    cfg = load_config()

    api_key = get_secret("OPENAI_API_KEY", "")
    sender = get_secret("SENDER_EMAIL", "")
    password = get_secret("SENDER_PASSWORD", "")
    recipients_raw = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")
    model = cfg.get("model", DEFAULT_MODEL)

    recipients = [x.strip() for x in recipients_raw.replace(";", ",").split(",") if x.strip()]
    expert_notes = cfg.get("auto_expert_notes", "")
    user_news = cfg.get("auto_user_news", "")

    report_date = datetime.now().strftime("%Y-%m-%d")
    result = run_all_pipeline(api_key, model, report_date, cfg, user_news, expert_notes)

    ok, msg = send_email_yagmail(sender, password, recipients, result["email_subject"], result["email_body"])
    log_email_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sender, recipients, result["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)

    save_market_history(report_date, result["morning_note"], result["closing_note"])
    save_ic_history(report_date, result["ic_note"])
    save_trade_ideas_history(report_date, result["trade_ideas"])


# =========================================================
# STREAMLIT UI
# =========================================================
def init_session():
    defaults = {
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


def main_streamlit():
    ensure_files()
    init_session()
    cfg = load_config()

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("TVAM@2026")

    api_key = get_secret("OPENAI_API_KEY", "")
    sender = get_secret("SENDER_EMAIL", "")
    sender_pw = get_secret("SENDER_PASSWORD", "")
    recipients_default = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")

    st.sidebar.title("⚙️ Cấu hình")

    if not api_key:
        st.sidebar.warning("Chưa cấu hình OPENAI_API_KEY trong secrets.")
    if not sender_pw:
        st.sidebar.warning("Chưa cấu hình SENDER_PASSWORD trong secrets.")

    model = st.sidebar.text_input("Model", value=cfg.get("model", DEFAULT_MODEL))
    report_date = str(st.sidebar.date_input("Ngày báo cáo"))

    house_view = st.sidebar.selectbox(
        "House View",
        ["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"],
        index=["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"].index(cfg.get("house_view", "Neutral"))
    )

    smart_mode = st.sidebar.checkbox("Smart mode", value=cfg.get("smart_mode", True))

    if sender:
        st.sidebar.caption(f"Email gửi đang dùng: {sender}")

    recipients = st.sidebar.text_area("Recipients", value=recipients_default, height=80)

    trade_ideas_count = st.sidebar.number_input(
        "Trade ideas count",
        min_value=3,
        max_value=10,
        value=int(cfg.get("trade_ideas_count", 7))
    )

    auto_send_after_run_all = st.sidebar.checkbox(
        "Run All xong tự gửi email",
        value=cfg.get("auto_send_after_run_all", False)
    )

    if st.sidebar.button("💾 Lưu config"):
        cfg["model"] = model
        cfg["house_view"] = house_view
        cfg["smart_mode"] = smart_mode
        cfg["default_recipients"] = recipients
        cfg["trade_ideas_count"] = int(trade_ideas_count)
        cfg["auto_send_after_run_all"] = auto_send_after_run_all
        save_config(cfg)
        st.sidebar.success("Đã lưu config.")

    st.session_state["user_news"] = st.text_area("Tin bổ sung / Additional notes", value=st.session_state.get("user_news", ""), height=100)
    st.session_state["expert_notes"] = st.text_area("Khuyến nghị chuyên gia / quỹ / Expert & fund recommendations", value=st.session_state.get("expert_notes", ""), height=140)

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

            if cfg.get("auto_send_after_run_all", False):
                recipients_list = [x.strip() for x in recipients.replace(";", ",").split(",") if x.strip()]
                ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
                log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK" if ok else "ERROR", "" if ok else msg)

            st.success("Đã chạy xong toàn bộ pipeline.")

    tabs = st.tabs(["Dashboard", "Notes", "Allocation", "Trade Ideas", "Email"])

    with tabs[0]:
        if not st.session_state["market_df"].empty:
            st.subheader("Market")
            st.dataframe(st.session_state["market_df"], use_container_width=True)
            st.text_area("Market Highlights", build_market_highlights(st.session_state["market_df"]), height=130)
            st.write(f"**Market Bias:** {get_market_bias(st.session_state['market_df'])}")

            st.markdown("### Thị trường Việt Nam / Vietnam Market Focus")
            vn_market = st.session_state["market_df"][
                st.session_state["market_df"]["Asset"].isin(["VNINDEX", "USD/VND (proxy)"])
            ]
            if not vn_market.empty:
                st.dataframe(vn_market, use_container_width=True)

            st.markdown("### Mã cổ phiếu Việt Nam được nhiều chuyên gia theo dõi / Recommended Vietnam Watchlist")
            vn_watch_df = fetch_vn_recommendation_watchlist()
            if not vn_watch_df.empty:
                st.dataframe(vn_watch_df, use_container_width=True)
            else:
                st.info("Chưa lấy được dữ liệu watchlist cổ phiếu Việt Nam.")

        if not st.session_state["news_df"].empty:
            vn_df, global_df = split_news_by_region(st.session_state["news_df"])
            a, b = st.columns(2)

            with a:
                st.markdown("#### Tin Việt Nam / Vietnam News")
                if not vn_df.empty:
                    vn_show = vn_df.copy()
                    vn_show["ArticleLink"] = vn_show["Link"].apply(
                        lambda x: f'<a href="{x}" target="_blank">Mở bài</a>' if pd.notna(x) and str(x).strip() else ""
                    )
                    show_cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in vn_show.columns]
                    st.write(vn_show[show_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("Không có tin Việt Nam.")

            with b:
                st.markdown("#### Tin quốc tế / Global News")
                if not global_df.empty:
                    global_show = global_df.copy()
                    global_show["ArticleLink"] = global_show["Link"].apply(
                        lambda x: f'<a href="{x}" target="_blank">Open article</a>' if pd.notna(x) and str(x).strip() else ""
                    )
                    show_cols = [c for c in ["Source", "Title", "AssetClass", "VNImpact", "ArticleLink"] if c in global_show.columns]
                    st.write(global_show[show_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("Không có tin quốc tế.")

            st.text_area("Signals / Tín hiệu", build_top_actionable_signals(st.session_state["news_df_ai"], 5), height=120)

    with tabs[1]:
        a, b, c = st.columns(3)
        with a:
            if st.button("Generate Morning Note", use_container_width=True):
                allocation_text = st.session_state.get("allocation_text", "")
                if not allocation_text and not st.session_state["market_df"].empty:
                    allocation_text = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                    st.session_state["allocation_text"] = allocation_text
                st.session_state["morning_note"] = generate_morning_note(
                    api_key, model, report_date, house_view,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["user_news"], st.session_state["expert_notes"], allocation_text
                )
        with b:
            if st.button("Generate Closing Note", use_container_width=True):
                allocation_text = st.session_state.get("allocation_text", "")
                if not allocation_text and not st.session_state["market_df"].empty:
                    allocation_text = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                    st.session_state["allocation_text"] = allocation_text
                st.session_state["closing_note"] = generate_closing_note(
                    api_key, model, report_date,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["expert_notes"], allocation_text
                )
        with c:
            if st.button("Generate IC Note", use_container_width=True):
                allocation_text = st.session_state.get("allocation_text", "")
                if not allocation_text and not st.session_state["market_df"].empty:
                    allocation_text = format_allocation(compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"]))
                    st.session_state["allocation_text"] = allocation_text
                st.session_state["ic_note"] = generate_ic_note(
                    api_key, model, report_date,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["expert_notes"], allocation_text
                )

        if st.session_state["morning_note"]:
            st.text_area("Morning Note (Song ngữ Việt - Anh)", st.session_state["morning_note"], height=380)

        if st.session_state["closing_note"]:
            st.text_area("Closing Note (Song ngữ Việt - Anh)", st.session_state["closing_note"], height=380)

        if st.session_state["ic_note"]:
            st.text_area("IC Note (Song ngữ Việt - Anh)", st.session_state["ic_note"], height=460)

    with tabs[2]:
        st.markdown("### Portfolio Allocation Engine")
        if st.button("Compute Allocation", use_container_width=True):
            allocation = compute_portfolio_allocation(st.session_state["market_df"], st.session_state["news_df_ai"])
            st.session_state["allocation_text"] = format_allocation(allocation)

        if st.session_state["allocation_text"]:
            st.text_area("Allocation Output", st.session_state["allocation_text"], height=280)

    with tabs[3]:
        st.markdown("### Trade Idea Scoring & Ranking")
        if st.button("Generate Ranked Trade Ideas", use_container_width=True):
            st.session_state["trade_ideas_df"] = generate_ranked_trade_ideas(
                st.session_state["market_df"], st.session_state["news_df_ai"], int(trade_ideas_count)
            )
            st.session_state["trade_ideas"] = format_trade_ideas_df(st.session_state["trade_ideas_df"])

        if isinstance(st.session_state.get("trade_ideas_df"), pd.DataFrame) and not st.session_state["trade_ideas_df"].empty:
            st.dataframe(st.session_state["trade_ideas_df"], use_container_width=True)

        if st.session_state["trade_ideas"]:
            st.text_area("Trade Ideas", st.session_state["trade_ideas"], height=360)
            st.text_area("Top Idea", extract_top_idea(st.session_state["trade_ideas"]), height=120)

    with tabs[4]:
        base_note = st.session_state["morning_note"] or st.session_state["closing_note"] or ""
        if st.button("Generate Email", use_container_width=True) and base_note:
            subject, body = build_email_bundle(
                api_key, model, report_date, "MANUAL",
                base_note,
                st.session_state.get("ic_note", ""),
                st.session_state.get("trade_ideas", ""),
                st.session_state.get("allocation_text", "")
            )
            st.session_state["email_subject"] = subject
            st.session_state["email_body"] = body

        st.session_state["email_subject"] = st.text_input("Subject / Tiêu đề", value=st.session_state.get("email_subject", ""))
        st.session_state["email_body"] = st.text_area("Body / Nội dung", value=st.session_state.get("email_body", ""), height=320)

        if st.button("Send Email Now", use_container_width=True):
            recipients_list = [x.strip() for x in recipients.replace(";", ",").split(",") if x.strip()]
            ok, msg = send_email_yagmail(sender, sender_pw, recipients_list, st.session_state["email_subject"], st.session_state["email_body"])
            if ok:
                st.success("Gửi email thành công")
                log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "OK", "")
            else:
                st.error(msg)
                log_email_send(report_date, sender, recipients_list, st.session_state["email_subject"], "ERROR", msg)


if __name__ == "__main__":
    if "--auto" in sys.argv:
        run_auto_job()
    else:
        main_streamlit()
