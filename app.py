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


APP_TITLE = "📊 Analyst Dashboard v5.6 - CIO Bilingual"
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
AUTO_RUN_LOG_FILE = DATA_DIR / "auto_run_log.txt"

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


def generate_with_openai(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1200) -> str:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return getattr(response, "output_text", "").strip()


def cached_ai_call(api_key: str, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int = 1200) -> str:
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
    title_l = (title or "").lower()
    summary_l = (summary or "").lower()
    text = f"{source_l} {title_l} {summary_l}"

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


def build_vn_global_news_brief(news_df: pd.DataFrame, vn_top_n: int = 3, global_top_n: int = 3) -> str:
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


def build_top_actionable_signals(news_df: pd.DataFrame, top_n: int = 3) -> str:
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


def fallback_morning_note(note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str) -> str:
    return f"""MORNING NOTE / BẢN TIN SÁNG - {note_date}

1. Bối cảnh thị trường / Market Backdrop
- House view: {bias}
- Thị trường cần thêm tín hiệu xác nhận trước khi nâng mạnh mức rủi ro.
- The market needs more confirmation before materially increasing risk.

2. Fixed Income
- Theo dõi lãi suất, lợi suất và định hướng chính sách.
- Monitor rates, yields and policy direction.

3. Equity
- Tập trung vào nhóm dẫn dắt, độ rộng và chất lượng dòng tiền.
- Focus on leadership groups, breadth and flow quality.

4. Commodity / FX
- Dầu, vàng, USD và tỷ giá là biến số quan trọng cho tactical positioning.
- Oil, gold, USD and FX remain key tactical variables.

5. Góc nhìn CIO / CIO View
- Ưu tiên allocation linh hoạt, chỉ tăng rủi ro khi xác nhận rõ hơn.
- Keep allocation flexible and add risk only on clearer confirmation.

6. Tin nổi bật / Key News
{build_vn_global_news_brief(news_df, 3, 3)}

7. Hành động đề xuất / Suggested Actions
{build_top_actionable_signals(news_df, 3)}

8. Ghi chú thêm / Additional Notes
{user_news[:300] if user_news else "- Không có. / None."}
"""


def fallback_closing_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
    return f"""CLOSING NOTE / BẢN TIN CUỐI NGÀY - {note_date}

1. Diễn biến cuối ngày / End-of-day Move
{build_market_highlights(market_df)}

2. Fixed Income
- Kiểm tra lại diễn biến lợi suất và implication cho duration.
- Reassess yield moves and implications for duration.

3. Equity
- Đánh giá thị trường có củng cố hay làm suy yếu view đầu ngày.
- Assess whether the market strengthened or weakened the morning view.

4. Commodity / FX
- Dầu, vàng, USD tiếp tục chi phối risk sentiment.
- Oil, gold and USD continue to shape risk sentiment.

5. Góc nhìn CIO / CIO View
- Chưa nên thay đổi allocation mạnh nếu tín hiệu xác nhận còn yếu.
- Avoid aggressive allocation shifts if confirmation remains weak.

6. Kế hoạch cho phiên tới / Plan for Next Session
- Theo dõi nhóm dẫn dắt, tỷ giá và biến động lãi suất.
- Monitor leadership, FX and rate volatility.
"""


def fallback_ic_note(note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    return f"""IC NOTE / GHI CHÚ IC - {note_date}

1. Fixed Income
- Cần theo dõi xu hướng lãi suất, lợi suất và duration stance.
- Monitor rate direction, yields and duration stance.

2. Equity
- Tập trung vào positioning, breadth và nhóm dẫn dắt.
- Focus on positioning, breadth and leadership.

3. Commodity / FX
- Dầu, vàng, USD có ảnh hưởng lớn tới tactical allocation.
- Oil, gold and USD matter for tactical allocation.

4. Nhận định chuyên gia / Expert and Fund Views
- {expert_notes[:500] if expert_notes else "Không có. / None."}

5. Góc nhìn CIO / CIO Deep Dive
- Asset allocation cần cân bằng giữa phòng thủ và tăng trưởng.
- Allocation should stay balanced between defense and growth.
- Chỉ nên tăng risk khi xác suất thắng cải thiện rõ rệt.
- Add risk only when the probability of success improves clearly.

6. Tin nổi bật / Key News
{build_vn_global_news_brief(news_df, 3, 3)}
"""


def fallback_trade_ideas(market_df: pd.DataFrame, news_df: pd.DataFrame, ideas_count: int = 5) -> str:
    vn_products = "\n".join([f"- {x}" for group in DEFAULT_VN_PRODUCTS.values() for x in group[:3]])
    return f"""TRADE IDEAS ƯU TIÊN VIỆT NAM / VIETNAM-FOCUSED TRADE IDEAS

1. VNINDEX / ETF Việt Nam
- Hướng hành động / Direction: Theo dõi mua khi tín hiệu cải thiện / Watch for tactical long on improving signals
- Luận điểm / Thesis: Đại diện tốt nhất cho xu hướng chung / Best proxy for broad market direction
- Rủi ro / Risk: Biến động ngắn hạn / Short-term volatility
- Score: 7
- Confidence: Medium

2. Nhóm ngân hàng Việt Nam / Vietnam banks
- Hướng hành động / Direction: Ưu tiên theo dõi / Positive watchlist
- Luận điểm / Thesis: Có thể dẫn dắt chỉ số / Potential market leadership
- Rủi ro / Risk: Áp lực lãi suất, tín dụng / Rate and credit pressure
- Score: 7
- Confidence: Medium

3. FPT / công nghệ Việt Nam
- Hướng hành động / Direction: Quan sát tích cực / Constructive watch
- Luận điểm / Thesis: Câu chuyện tăng trưởng rõ hơn mặt bằng chung / Clearer growth profile
- Rủi ro / Risk: Định giá / Valuation
- Score: 8
- Confidence: Medium

4. USD/VND
- Hướng hành động / Direction: Theo dõi phòng thủ / Defensive monitor
- Luận điểm / Thesis: Tỷ giá ảnh hưởng tâm lý và định vị rủi ro / FX shapes sentiment and risk positioning
- Rủi ro / Risk: Biến động chính sách / Policy shifts
- Score: 6
- Confidence: Medium

5. Vàng / dầu tác động tới Việt Nam
- Hướng hành động / Direction: Theo dõi chiến thuật / Tactical monitoring
- Luận điểm / Thesis: Tác động tới lạm phát và sentiment / Impacts inflation and sentiment
- Rủi ro / Risk: Nhiễu ngắn hạn / Short-term noise
- Score: 5
- Confidence: Medium

Sản phẩm Việt Nam tham khảo / Vietnam product universe:
{vn_products}

Tín hiệu / Signals:
{build_top_actionable_signals(news_df, min(ideas_count, 3))}
"""


def generate_morning_note(api_key: str, model: str, note_date: str, bias: str, market_df: pd.DataFrame, news_df: pd.DataFrame, user_news: str) -> str:
    system_prompt = """You are a market strategist supporting a CIO in Vietnam.
Write a STRICTLY BILINGUAL Vietnamese-English Morning Note.
Every heading must be bilingual. Every bullet must contain both Vietnamese and English.
"""
    user_prompt = f"""
Date: {note_date}
House view: {bias}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 3, 3)}

SIGNALS:
{build_top_actionable_signals(news_df, 3)}

USER NOTES:
{normalize_text(user_news)[:300]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1400)
    return out if out else fallback_morning_note(note_date, bias, market_df, news_df, user_news)


def generate_closing_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
    system_prompt = """You are an end-of-day strategist supporting a CIO in Vietnam.
Write a STRICTLY BILINGUAL Vietnamese-English Closing Note.
Every heading must be bilingual. Every bullet must contain both Vietnamese and English.
"""
    user_prompt = f"""
Date: {note_date}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 3, 3)}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1400)
    return out if out else fallback_closing_note(note_date, market_df, news_df)


def generate_ic_note(api_key: str, model: str, note_date: str, market_df: pd.DataFrame, news_df: pd.DataFrame, expert_notes: str) -> str:
    system_prompt = """You are a PM/strategist supporting a CIO in Vietnam.
Write a STRICTLY BILINGUAL Vietnamese-English IC/PM Note.
Every heading must be bilingual. Every bullet must contain both Vietnamese and English.
"""
    user_prompt = f"""
Date: {note_date}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 3, 3)}

EXPERT NOTES:
{normalize_text(expert_notes)[:800]}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1500)
    return out if out else fallback_ic_note(note_date, market_df, news_df, expert_notes)


def generate_trade_ideas(api_key: str, model: str, market_df: pd.DataFrame, news_df: pd.DataFrame, ideas_count: int = 5) -> str:
    if news_df.empty:
        return fallback_trade_ideas(market_df, news_df, ideas_count)

    vn_products_text = "\n".join(f"- {group}: {', '.join(items)}" for group, items in DEFAULT_VN_PRODUCTS.items())
    system_prompt = """You are a CIO/PM in Vietnam.
Write Vietnam-focused trade ideas in bilingual Vietnamese-English format.
Prioritize Vietnamese assets and products.
Include Asset/Product, Direction, Thesis, Risk, Score, Confidence.
Max 5 ranked ideas.
"""
    user_prompt = f"""
Ideas count: {ideas_count}

MARKET:
{build_market_highlights(market_df)}

NEWS:
{build_vn_global_news_brief(news_df, 3, 3)}

SIGNALS:
{build_top_actionable_signals(news_df, 3)}

VIETNAM PRODUCT UNIVERSE:
{vn_products_text}
"""
    out = cached_ai_call(api_key, model, system_prompt, user_prompt, 1000)
    return out if out else fallback_trade_ideas(market_df, news_df, ideas_count)


def extract_top_idea(trade_ideas_text: str) -> str:
    text = normalize_text(trade_ideas_text)
    if not text:
        return "Chưa có ý tưởng nổi bật / No top idea available."
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks:
        return blocks[0]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return "\n".join(lines[:6]) if lines else "Chưa có ý tưởng nổi bật / No top idea available."


def generate_email_body(api_key: str, model: str, core_note: str, report_date: str, run_label: str) -> Tuple[str, str]:
    subject = f"[{run_label}] Cập nhật thị trường | Market Update - {report_date}"
    system_prompt = """Write a STRICTLY BILINGUAL Vietnamese-English market update email.
Greeting must be bilingual. Every major paragraph must contain both Vietnamese and English.
End with:
Trân trọng.
Best regards.
"""
    out = cached_ai_call(api_key, model, system_prompt, core_note[:2800], 900)
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

    morning_note = generate_morning_note(api_key, model, report_date, cfg.get("house_view", "Neutral"), market_df, news_df_ai, user_news)
    closing_note = generate_closing_note(api_key, model, report_date, market_df, news_df_ai)
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
    house_view = cfg.get("house_view", "Neutral")
    sender = get_runtime_value(cfg.get("default_sender_email", ""), "SENDER_EMAIL")
    password = get_runtime_value(cfg.get("default_sender_password", ""), "SENDER_PASSWORD")
    recipients_raw = get_runtime_value(cfg.get("default_recipients", ""), "DEFAULT_RECIPIENTS")
    recipients = [x.strip() for x in recipients_raw.replace(";", ",").split(",") if x.strip()]
    smart_mode = bool(cfg.get("smart_mode", True))
    trade_ideas_count = int(cfg.get("trade_ideas_count", 5))
    expert_notes = cfg.get("auto_expert_notes", "")
    user_news = cfg.get("auto_user_news", "")

    now = datetime.now()
    report_date = now.strftime("%Y-%m-%d")
    run_label = "08:00" if now.hour < 12 else "16:00"

    append_auto_log("START auto run")

    market_df = fetch_market_snapshot(DEFAULT_TICKERS)
    news_df = fetch_rss_news(RSS_FEEDS, max_per_feed=6)
    news_df_ai = filter_news_for_ai(news_df, smart_mode=smart_mode)

    core_note = generate_morning_note(api_key, model, report_date, house_view, market_df, news_df_ai, user_news) if run_label == "08:00" else generate_closing_note(api_key, model, report_date, market_df, news_df_ai)
    ic_note = generate_ic_note(api_key, model, report_date, market_df, news_df_ai, expert_notes)
    trade_ideas = generate_trade_ideas(api_key, model, market_df, news_df_ai, trade_ideas_count)

    subject, body = build_email_bundle(api_key, model, report_date, run_label, core_note, ic_note, trade_ideas)

    ok, msg = send_email_yagmail(sender, password, recipients, subject, body)
    log_email_send(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sender, recipients, subject, "OK" if ok else "ERROR", "" if ok else msg)

    save_market_history(report_date, core_note if run_label == "08:00" else "", core_note if run_label == "16:00" else "")
    save_ic_history(report_date, ic_note)
    save_trade_ideas_history(report_date, trade_ideas)

    append_auto_log(f"END auto run | {run_label}")


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
    st.caption("Bản deploy Streamlit Cloud | song ngữ Việt-Anh | có link bài báo trong bảng")

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
    st.session_state["expert_notes"] = st.text_area("Expert notes / Ghi chú chuyên gia", value=st.session_state.get("expert_notes", ""), height=90)

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

            st.text_area("Signals / Tín hiệu", build_top_actionable_signals(st.session_state["news_df_ai"], 3), height=100)

    with tabs[1]:
        a, b, c = st.columns(3)
        with a:
            if st.button("Generate Morning Note", use_container_width=True):
                st.session_state["morning_note"] = generate_morning_note(api_key, model, report_date, house_view, st.session_state["market_df"], st.session_state["news_df_ai"], st.session_state["user_news"])
        with b:
            if st.button("Generate Closing Note", use_container_width=True):
                st.session_state["closing_note"] = generate_closing_note(api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"])
        with c:
            if st.button("Generate IC Note", use_container_width=True):
                st.session_state["ic_note"] = generate_ic_note(api_key, model, report_date, st.session_state["market_df"], st.session_state["news_df_ai"], st.session_state["expert_notes"])

        if st.session_state["morning_note"]:
            st.text_area("Morning Note (Song ngữ Việt - Anh)", st.session_state["morning_note"], height=340)
        if st.session_state["closing_note"]:
            st.text_area("Closing Note (Song ngữ Việt - Anh)", st.session_state["closing_note"], height=340)
        if st.session_state["ic_note"]:
            st.text_area("IC Note (Song ngữ Việt - Anh)", st.session_state["ic_note"], height=380)

    with tabs[2]:
        st.markdown("#### Trade Ideas ưu tiên sản phẩm / tài sản ở Việt Nam")
        st.write(", ".join([x for group in DEFAULT_VN_PRODUCTS.values() for x in group[:6]]))

        if st.button("Generate Trade Ideas", use_container_width=True):
            st.session_state["trade_ideas"] = generate_trade_ideas(api_key, model, st.session_state["market_df"], st.session_state["news_df_ai"], int(trade_ideas_count))

        if st.session_state["trade_ideas"]:
            st.text_area("Trade Ideas", st.session_state["trade_ideas"], height=340)
            st.text_area("Top Idea", extract_top_idea(st.session_state["trade_ideas"]), height=110)

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
        st.session_state["email_body"] = st.text_area("Body / Nội dung", value=st.session_state.get("email_body", ""), height=280)

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