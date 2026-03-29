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


APP_TITLE = "📊 Analyst Dashboard - TVAM"
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
            "openai_api_key": "",
            "model": DEFAULT_MODEL,
            "house_view": "Neutral",
            "default_sender_email": "",
            "default_sender_password": "",
            "default_recipients": "",
            "smart_mode": True,
            "enable_trade_ideas": True,
            "enable_closing_note": True,
            "auto_send_after_run_all": False,
            "trade_ideas_count": 5,
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
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_runtime_value(config_value: str, secret_name: str) -> str:
    secret_val = get_secret(secret_name, "")
    return secret_val if secret_val else (config_value or "")


def generate_with_openai(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1400) -> str:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return getattr(response, "output_text", "").strip()


def cached_ai_call(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1400) -> str:
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


def build_expert_fund_summary(user_expert_notes: str = "") -> str:
    base = """
- SSI: Ưu tiên theo dõi large caps, câu chuyện nâng hạng và dòng vốn quay lại / Focus on large caps, upgrade story and returning flows.
- VNDirect: Theo dõi biến động lãi suất, tỷ giá, nhóm dẫn dắt trong nước / Monitor rates, FX and domestic leadership.
- Quỹ / Funds: Ưu tiên allocation linh hoạt, quản trị risk chặt chẽ / Prefer flexible allocation and tight risk management.
"""
    if normalize_text(user_expert_notes):
        base += "\n- Ghi chú thêm / Additional expert notes:\n" + normalize_text(user_expert_notes)[:1000]
    return base


def fallback_morning_note(note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str) -> str:
    return f"""MORNING NOTE / BẢN TIN SÁNG - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Theo dõi lợi suất và kỳ vọng chính sách vì dầu và USD đang ảnh hưởng mạnh đến định vị duration.
- Monitor yields and policy expectations as oil and USD are materially influencing duration positioning.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tỷ giá, thanh khoản hệ thống và định hướng điều hành lãi suất.
- Monitor FX, system liquidity, and domestic rate policy direction.

3. Global Equity / Equity toàn cầu
- Tâm lý cổ phiếu toàn cầu phụ thuộc vào risk-on/risk-off, dầu và lợi suất.
- Global equity sentiment depends on risk-on/risk-off, oil and yields.

4. Vietnam Equity / Equity Việt Nam
- Tập trung vào nhóm dẫn dắt, thanh khoản và dòng vốn ngoại.
- Focus on leadership sectors, liquidity and foreign flows.

5. Commodity / FX
- Dầu, vàng, USD và USD/VND là biến số chiến thuật quan trọng.
- Oil, gold, USD and USD/VND are key tactical variables.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. CIO / IC / PM View and Actions
- House view: {bias}
- Ưu tiên allocation linh hoạt, chưa tăng mạnh rủi ro nếu xác nhận xu hướng chưa rõ.
- Keep allocation flexible and avoid adding aggressive risk without clearer trend confirmation.

8. Key news / Tin nổi bật
{build_vn_global_news_brief(news_df, 4, 4)}

9. Additional notes / Ghi chú thêm
{user_news[:500] if user_news else "- Không có. / None."}
"""


def fallback_closing_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    return f"""CLOSING NOTE / BẢN TIN CUỐI NGÀY - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Đánh giá lại biến động lợi suất và implication cho duration.
- Reassess yield moves and implications for duration.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Theo dõi tín hiệu tỷ giá và thanh khoản nội địa.
- Monitor domestic FX and liquidity signals.

3. Global Equity / Equity toàn cầu
- Đánh giá liệu risk sentiment đang cải thiện hay xấu đi.
- Assess whether risk sentiment is improving or deteriorating.

4. Vietnam Equity / Equity Việt Nam
- Kiểm tra độ rộng, thanh khoản và nhóm dẫn dắt.
- Check breadth, liquidity and leadership sectors.

5. Commodity / FX
- Dầu, vàng và USD tiếp tục là biến số lớn.
- Oil, gold and USD remain major variables.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Plan for next session / Kế hoạch phiên tới
- Giữ kỷ luật position sizing và theo dõi tín hiệu xác nhận.
- Maintain disciplined position sizing and monitor confirmation signals.
"""


def fallback_ic_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    return f"""IC NOTE / GHI CHÚ IC - {note_date}

1. Global Fixed Income / Fixed Income toàn cầu
- Duration view nên giữ trung lập cho tới khi biến động dầu và lợi suất dịu lại.
- Duration stance should stay neutral until oil and yield volatility eases.

2. Vietnam Fixed Income / Fixed Income Việt Nam
- Ưu tiên theo dõi tỷ giá, thanh khoản và đường cong lợi suất.
- Focus on FX, liquidity and local yield curve.

3. Global Equity / Equity toàn cầu
- Positioning nên linh hoạt theo mức độ risk-on/risk-off.
- Positioning should remain flexible based on risk-on/risk-off conditions.

4. Vietnam Equity / Equity Việt Nam
- Ưu tiên large caps, nhóm dẫn dắt và câu chuyện nâng hạng.
- Prefer large caps, leadership groups and the upgrade story.

5. Commodity / FX
- Dầu, vàng, USD và USD/VND tác động trực tiếp đến tactical allocation.
- Oil, gold, USD and USD/VND directly affect tactical allocation.

6. Expert / Fund Recommendations Summary
{build_expert_fund_summary(expert_notes)}

7. Portfolio implication / Hàm ý danh mục
- Overweight có chọn lọc ở equity Việt Nam, neutral duration, theo dõi commodity shock.
- Selective overweight in Vietnam equities, neutral duration, monitor commodity shock.
"""


def generate_morning_note(api_key: str, model: str, note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str, expert_notes: str) -> str:
    system_prompt = """You are a strategist supporting a CIO / IC / portfolio manager in Vietnam.

Write a STRICTLY BILINGUAL Vietnamese-English Morning Note.

Required structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. CIO / IC / PM View and Actions

Rules:
- Every heading must be bilingual.
- Every bullet must contain both Vietnamese and English.
- Focus on the latest market implications.
- Mention allocation implication, duration view, equity positioning, commodity implication.
"""
    user_prompt = f"""
Date: {note_date}
House view: {bias}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

SIGNALS:
{build_top_actionable_signals(news_df, 5)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}

USER NOTES:
{normalize_text(user_news)[:500]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1700)
    return out if out else fallback_morning_note(note_date, bias, market_df, news_df, user_news, expert_notes)


def generate_closing_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    system_prompt = """You are an end-of-day strategist supporting a CIO / IC / portfolio manager in Vietnam.

Write a STRICTLY BILINGUAL Vietnamese-English Closing Note.

Required structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. Changes vs Morning / What Changed and Next Session Plan

Rules:
- Every heading must be bilingual.
- Every bullet must contain both Vietnamese and English.
"""
    user_prompt = f"""
Date: {note_date}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1700)
    return out if out else fallback_closing_note(note_date, market_df, news_df, expert_notes)


def generate_ic_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    system_prompt = """You are a PM / strategist supporting a CIO / IC / portfolio manager in Vietnam.

Write a STRICTLY BILINGUAL Vietnamese-English IC / PM Note.

Required structure:
1. Global Fixed Income / Fixed Income toàn cầu
2. Vietnam Fixed Income / Fixed Income Việt Nam
3. Global Equity / Equity toàn cầu
4. Vietnam Equity / Equity Việt Nam
5. Commodity / FX
6. Expert / Fund Recommendations Summary
7. Portfolio implication / Suggested IC actions

Rules:
- Every heading must be bilingual.
- Every bullet must contain both Vietnamese and English.
- Explicitly mention overweight / neutral / underweight when appropriate.
"""
    user_prompt = f"""
Date: {note_date}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

EXPERT / FUND VIEWS:
{build_expert_fund_summary(expert_notes)}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1800)
    return out if out else fallback_ic_note(note_date, market_df, news_df, expert_notes)


def generate_trade_ideas(api_key: str, model: str, market_df: pd.DataFrame, news_df: pd.DataFrame, ideas_count: int = 5) -> str:
    if news_df.empty:
        return "Chưa có dữ liệu trade ideas / No trade idea data."

    vn_products_text = "\n".join(f"- {group}: {', '.join(items)}" for group, items in DEFAULT_VN_PRODUCTS.items())
    system_prompt = """You are a CIO / PM in Vietnam.

Write Vietnam-focused trade ideas in bilingual Vietnamese-English format.

Requirements:
- prioritize Vietnamese assets and products
- include: Asset/Product, Direction, Thesis, Risk, Score, Confidence
- max 5 ranked ideas
"""
    user_prompt = f"""
Ideas count: {ideas_count}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 4, 4)}

SIGNALS:
{build_top_actionable_signals(news_df, 5)}

VIETNAM PRODUCT UNIVERSE:
{vn_products_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1200)
    if out:
        return out

    return """TRADE IDEAS / Ý TƯỞNG ĐẦU TƯ

1. VNINDEX / ETF Việt Nam
- Direction / Hướng hành động: Theo dõi mua khi tín hiệu cải thiện / Watch for tactical long on improving signals
- Thesis / Luận điểm: Đại diện tốt cho xu hướng chung / Good proxy for broad market direction
- Risk / Rủi ro: Biến động ngắn hạn / Short-term volatility
- Score: 7
- Confidence: Medium

2. Nhóm ngân hàng Việt Nam / Vietnam banks
- Direction / Hướng hành động: Theo dõi tích cực / Positive watch
- Thesis / Luận điểm: Có thể dẫn dắt thị trường / Potential market leadership
- Risk / Rủi ro: Áp lực tín dụng và lãi suất / Credit and rate pressure
- Score: 7
- Confidence: Medium
"""


def extract_top_idea(trade_ideas_text: str) -> str:
    text = normalize_text(trade_ideas_text)
    if not text:
        return "Chưa có ý tưởng nổi bật / No top idea available."
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks:
        return blocks[0]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return "\n".join(lines[:8]) if lines else "Chưa có ý tưởng nổi bật / No top idea available."


def generate_email_body(api_key: str, model: str, core_note: str, report_date: str, run_label: str) -> Tuple[str, str]:
    subject = f"[{run_label}] Cập nhật thị trường | Market Update - {report_date}"
    system_prompt = """Write a STRICTLY BILINGUAL Vietnamese-English market update email.

Rules:
- Greeting must be bilingual.
- Every major paragraph must contain both Vietnamese and English.
- End with:
Trân trọng.
Best regards.
"""
    out = cached_ai_call(api_key, model, system_prompt, core_note[:3000], 1000)
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
    append_csv_row(MARKET_HISTORY_FILE, {"Date": note_date, "MorningNote": morning_note[:4000], "ClosingNote": closing_note[:4000]})


def save_ic_history(note_date: str, ic_note: str) -> None:
    append_csv_row(IC_HISTORY_FILE, {"Date": note_date, "ICNote": ic_note[:4000]})


def save_trade_ideas_history(note_date: str, trade_ideas: str) -> None:
    append_csv_row(TRADE_IDEAS_HISTORY_FILE, {"Date": note_date, "TradeIdeas": trade_ideas[:4000]})


def build_email_bundle(api_key: str, model: str, report_date: str, run_label: str, core_note: str, ic_note: str, trade_ideas: str) -> Tuple[str, str]:
    subject, body = generate_email_body(api_key, model, core_note, report_date, run_label)
    top_idea = extract_top_idea(trade_ideas)
    if top_idea:
        body += "\n\n=== Ý tưởng đầu tư nổi bật | Top Trade Idea ===\n" + top_idea
    if trade_ideas:
        body += "\n\n=== Danh sách Trade Ideas | Full Trade Ideas ===\n" + trade_ideas
    if ic_note:
        body += "\n\n=== IC Note ===\n" + ic_note
    return subject, body


def run_all_pipeline(api_key: str, model: str, report_date: str, cfg: dict, user_news: str, expert_notes: str):
    market_df = fetch_market_snapshot(DEFAULT_TICKERS)
    news_df = fetch_rss_news(RSS_FEEDS, max_per_feed=6)
    news_df_ai = filter_news_for_ai(news_df, smart_mode=cfg.get("smart_mode", True))

    morning_note = generate_morning_note(api_key, model, report_date, cfg.get("house_view", "Neutral"), market_df, news_df_ai, user_news, expert_notes)
    closing_note = generate_closing_note(api_key, model, report_date, market_df, news_df_ai, expert_notes)
    ic_note = generate_ic_note(api_key, model, report_date, market_df, news_df_ai, expert_notes)
    trade_ideas = generate_trade_ideas(api_key, model, market_df, news_df_ai, int(cfg.get("trade_ideas_count", 5)))
    email_subject, email_body = build_email_bundle(api_key, model, report_date, "MANUAL", morning_note, ic_note, trade_ideas)

    return {
        "market_df": market_df,
        "news_df": news_df,
        "news_df_ai": news_df_ai,
        "morning_note": morning_note,
        "closing_note": closing_note,
        "ic_note": ic_note,
        "trade_ideas": trade_ideas,
        "email_subject": email_subject,
        "email_body": email_body,
    }


def run_auto_job() -> None:
    ensure_files()
    cfg = load_config()

    api_key = get_runtime_value(cfg.get("openai_api_key", ""), "OPENAI_API_KEY")
    model = cfg.get("model", DEFAULT_MODEL)
    sender = get_runtime_value(cfg.get("default_sender_email", ""), "SENDER_EMAIL")
    password = get_runtime_value(cfg.get("default_sender_password", ""), "SENDER_PASSWORD")
    recipients_raw = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")
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


def init_session():
    defaults = {
        "market_df": pd.DataFrame(),
        "news_df": pd.DataFrame(),
        "news_df_ai": pd.DataFrame(),
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

    api_key_default = get_runtime_value(cfg.get("openai_api_key", ""), "OPENAI_API_KEY")
    sender_default = get_runtime_value(cfg.get("default_sender_email", ""), "SENDER_EMAIL")
    sender_pw_default = get_runtime_value(cfg.get("default_sender_password", ""), "SENDER_PASSWORD")
    recipients_default = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")

    st.sidebar.title("⚙️ Cấu hình")
    api_key = st.sidebar.text_input("OpenAI API Key", value=api_key_default, type="password")
    model = st.sidebar.text_input("Model", value=cfg.get("model", DEFAULT_MODEL))
    report_date = str(st.sidebar.date_input("Ngày báo cáo"))
    house_view = st.sidebar.selectbox(
        "House View",
        ["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"],
        index=["Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Bearish"].index(cfg.get("house_view", "Neutral"))
    )
    smart_mode = st.sidebar.checkbox("Smart mode", value=cfg.get("smart_mode", True))
    sender = st.sidebar.text_input("Sender", value=sender_default)
    sender_pw = st.sidebar.text_input("App Password", value=sender_pw_default, type="password")
    recipients = st.sidebar.text_area("Recipients", value=recipients_default, height=80)
    trade_ideas_count = st.sidebar.number_input("Trade ideas count", min_value=1, max_value=10, value=int(cfg.get("trade_ideas_count", 5)))
    auto_send_after_run_all = st.sidebar.checkbox("Run All xong tự gửi email", value=cfg.get("auto_send_after_run_all", False))

    if st.sidebar.button("💾 Lưu config"):
        cfg["openai_api_key"] = api_key
        cfg["model"] = model
        cfg["house_view"] = house_view
        cfg["smart_mode"] = smart_mode
        cfg["default_sender_email"] = sender
        cfg["default_sender_password"] = sender_pw
        cfg["default_recipients"] = recipients
        cfg["trade_ideas_count"] = int(trade_ideas_count)
        cfg["auto_send_after_run_all"] = auto_send_after_run_all
        save_config(cfg)
        st.sidebar.success("Đã lưu config.")

    st.session_state["user_news"] = st.text_area("Tin bổ sung / Additional notes", value=st.session_state.get("user_news", ""), height=90)
    st.session_state["expert_notes"] = st.text_area("Khuyến nghị chuyên gia / quỹ / Expert & fund recommendations", value=st.session_state.get("expert_notes", ""), height=120)

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

    tabs = st.tabs(["Dashboard", "Notes", "Trade Ideas", "Email"])

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
                st.session_state["morning_note"] = generate_morning_note(
                    api_key, model, report_date, house_view,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["user_news"], st.session_state["expert_notes"]
                )
        with b:
            if st.button("Generate Closing Note", use_container_width=True):
                st.session_state["closing_note"] = generate_closing_note(
                    api_key, model, report_date,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["expert_notes"]
                )
        with c:
            if st.button("Generate IC Note", use_container_width=True):
                st.session_state["ic_note"] = generate_ic_note(
                    api_key, model, report_date,
                    st.session_state["market_df"], st.session_state["news_df_ai"],
                    st.session_state["expert_notes"]
                )

        if st.session_state["morning_note"]:
            st.text_area("Morning Note (Song ngữ Việt - Anh)", st.session_state["morning_note"], height=360)
        if st.session_state["closing_note"]:
            st.text_area("Closing Note (Song ngữ Việt - Anh)", st.session_state["closing_note"], height=360)
        if st.session_state["ic_note"]:
            st.text_area("IC Note (Song ngữ Việt - Anh)", st.session_state["ic_note"], height=420)

    with tabs[2]:
        st.markdown("#### Trade Ideas ưu tiên sản phẩm / tài sản ở Việt Nam")
        st.write(", ".join([x for group in DEFAULT_VN_PRODUCTS.values() for x in group[:6]]))

        if st.button("Generate Trade Ideas", use_container_width=True):
            st.session_state["trade_ideas"] = generate_trade_ideas(
                api_key, model, st.session_state["market_df"], st.session_state["news_df_ai"], int(trade_ideas_count)
            )

        if st.session_state["trade_ideas"]:
            st.text_area("Trade Ideas", st.session_state["trade_ideas"], height=340)
            st.text_area("Top Idea", extract_top_idea(st.session_state["trade_ideas"]), height=120)

    with tabs[3]:
        base_note = st.session_state["morning_note"] or st.session_state["closing_note"] or ""
        if st.button("Generate Email", use_container_width=True) and base_note:
            subject, body = build_email_bundle(
                api_key, model, report_date, "MANUAL",
                base_note,
                st.session_state.get("ic_note", ""),
                st.session_state.get("trade_ideas", "")
            )
            st.session_state["email_subject"] = subject
            st.session_state["email_body"] = body

        st.session_state["email_subject"] = st.text_input("Subject / Tiêu đề", value=st.session_state.get("email_subject", ""))
        st.session_state["email_body"] = st.text_area("Body / Nội dung", value=st.session_state.get("email_body", ""), height=300)

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