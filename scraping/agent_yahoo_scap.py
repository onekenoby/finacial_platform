import os
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from ollama import chat
from datetime import datetime
import hashlib
from tqdm import tqdm

# Project Paths [cite: 342, 344]
BASE_DIR = r"E:\Dev\FinancialAI"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_ingestion", "INBOX", "TIER_C_NEWS", "MARKET_NEWS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multimodal Engine [cite: 50, 57]
MODEL_NAME = "gemma3:12b"

# --- SYSTEM PROMPTS (ENGLISH) [cite: 253, 344] ---
AGENT_SYSTEM_PROMPT = """
You are a Senior Financial Market Analyst at Fin-Studio. 
Your task is to analyze raw web content and determine if it contains high-value market intelligence.
Focus on macroeconomic events, market trends, volatility drivers (VIX), and sentiment shifts.
"""

ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following web content for the ticker: {ticker}.

### DECISION CRITERIA:
1. If the content is spam, ads, clickbait, or lacks numerical data/meaningful financial insight, respond ONLY with "SKIP".
2. If relevant, generate a structured MARKDOWN file.

### OUTPUT REQUIREMENTS:
- **YAML Frontmatter**: Include source, ticker, date, sentiment (Bullish/Bearish/Neutral), and macro_impact.
- **Visual Reconstruction**: If the text describes technical levels, reconstruct them using Markdown tables.
- **Entities**: Identify key organizations mentioned.

### CONTENT TO ANALYZE:
Title: {title}
Ticker: {ticker}
Raw Text: {raw_text}
"""

def get_ticker_news(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.news or []
    except Exception:
        return []

def agent_process_and_save(news_item, ticker):
    # Bug Fix: Ensure title is always a string 
    title = str(news_item.get('title') or "Untitled News")
    url = news_item.get('link') or news_item.get('content', {}).get('canonicalUrl', {}).get('url')
    
    # Phase 1: Ingestion
    if not url or not url.startswith('http'):
        raw_text = str(news_item.get('summary') or "No summary available")
    else:
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            for s in soup(['script', 'style', 'nav', 'footer']): s.decompose()
            raw_text = soup.get_text(separator=' ', strip=True)[:10000]
        except Exception:
            return "error"

    # Phase 2: AI Interpretation [cite: 11, 65, 344]
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
                {'role': 'user', 'content': ANALYSIS_PROMPT_TEMPLATE.format(
                    ticker=ticker, title=title, raw_text=raw_text
                )}
            ],
            options={'temperature': 0.1, 'num_ctx': 16384}
        )
        
        # Bug Fix: Handle empty LLM response 
        content_out = response.get('message', {}).get('content')
        if content_out is None:
            return "error"

        if "SKIP" in content_out[:10].upper():
            return "skipped"

        # Phase 3: Storage 
        # Bug Fix: Hash title only if it's a valid string 
        file_hash = hashlib.md5(title.encode('utf-8', errors='ignore')).hexdigest()[:10]
        filename = f"NEWS_{ticker}_{file_hash}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_out)
        return "saved"
    except Exception as e:
        print(f"   ⚠️ Internal Processing Error: {e}")
        return "error"

def run_scraping_session(tickers):
    ticker_pbar = tqdm(tickers, desc="📈 Market Focus", unit="ticker")
    for ticker in ticker_pbar:
        ticker_pbar.set_postfix({"current": ticker})
        news_list = get_ticker_news(ticker)
        
        if not news_list: continue
            
        news_pbar = tqdm(news_list, desc=f"  └─ {ticker}", unit="news", leave=False)
        for item in news_pbar:
            result = agent_process_and_save(item, ticker)
            news_pbar.set_postfix({"status": result})

if __name__ == "__main__":
    print(f"🚀 Fin-Studio Scraper: Targeting {MODEL_NAME}")
    market_focus = ["^GSPC", "^VIX", "EURUSD=X", "BTC-USD", "GC=F", "CL=F"] 
    run_scraping_session(market_focus)