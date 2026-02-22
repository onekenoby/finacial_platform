import os
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from ollama import chat
from datetime import datetime
import hashlib
from tqdm import tqdm

# Project Paths
BASE_DIR = r"E:\Dev\FinancialAI"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_ingestion", "INBOX", "TIER_C_NEWS", "MARKET_NEWS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multimodal Engine (Vision + Text)
MODEL_NAME = "ministral-3:8b"

# --- SYSTEM PROMPTS (ENGLISH) ---
AGENT_SYSTEM_PROMPT = """
You are a Senior Financial Market Analyst, a Computer Vision Expert, and a Quantitative Engineer at Fin-Studio. 
Your task is to analyze BOTH the raw web text and any attached chart/image to extract high-value market intelligence, data, and mathematical models.
"""

ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following web content and the attached image (if present) for the ticker: {ticker}.

### DECISION CRITERIA:
1. If the content is spam, ads, clickbait, or lacks meaningful financial insight, respond ONLY with "SKIP".
2. If relevant, generate a structured MARKDOWN file.

### OUTPUT REQUIREMENTS:
- **YAML Frontmatter**: Include source, ticker, date, sentiment (Bullish/Bearish/Neutral), and macro_impact.
- **Visual Analysis (CRITICAL)**: If an image is provided, identify if it's a chart or table. Extract key financial data, describe visual trends (support/resistance, crossovers), and reconstruct the data points into a clear Markdown table using `| col | col |`.
- **Mathematical Formulas (QUANTITATIVE)**: If the text or the image contains formal mathematical equations, financial formulas, or pricing models (e.g., CAPM, Black-Scholes), transcribe them EXACTLY using standard LaTeX formatting enclosed in double dollars for blocks (e.g., `$$ WACC = ... $$`) or single dollars for inline. 
  *Rule*: Do NOT use LaTeX for simple currency values or basic percentages (e.g., write "$50" or "5%", not `$\$50$`).
- **Textual Synthesis**: Summarize the article's core thesis and identify key entities.

### CONTENT TO ANALYZE:
Title: {title}
Ticker: {ticker}
Raw Text: {raw_text}
"""

def get_ticker_news(ticker):
    """
    Recupera la lista delle notizie più recenti per un dato ticker utilizzando le API di Yahoo Finance.
    
    Args:
        ticker (str): Il simbolo del mercato (es. '^GSPC', 'AAPL').
        
    Returns:
        list: Una lista di dizionari contenenti i metadati delle news (link, titolo, ecc.).
              Ritorna una lista vuota in caso di errore.
    """
    try:
        t = yf.Ticker(ticker)
        return t.news or []
    except Exception:
        return []

def agent_process_and_save(news_item, ticker):
    """
    Core engine dell'agente. Elabora una singola notizia in tre fasi:
    1. Scraping: Scarica l'HTML pulito e cerca di estrarre l'immagine/grafico principale dell'articolo.
    2. AI Interpretation: Invia testo e immagine a Ministral (multimodale) per analisi e filtraggio.
    3. Storage: Salva il report in un file Markdown con naming convention basata su Timestamp.
    """
    title = str(news_item.get('title') or "Untitled News")
    url = news_item.get('link') or news_item.get('content', {}).get('canonicalUrl', {}).get('url')
    
    # --- NUOVA RIGA: Stampa l'URL a monitor in modo compatibile con le progress bar ---
    if url:
        tqdm.write(f"   🔗 Estrazione da: {url}")
    # ---------------------------------------------------------------------------------

    raw_text = ""
    image_bytes = None

    # Phase 1: Ingestion (Testo + Immagini)
    if not url or not url.startswith('http'):
        raw_text = str(news_item.get('summary') or "No summary available")
    else:
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ricerca dell'immagine principale (Target specifico per layout Yahoo Finance)
            img_tag = soup.find('img', {'class': 'caas-img'}) or soup.find('img')
            if img_tag and img_tag.get('src'):
                img_url = img_tag.get('src')
                if img_url.startswith('http'):
                    try:
                        img_response = requests.get(img_url, timeout=5)
                        if img_response.status_code == 200:
                            image_bytes = img_response.content
                    except Exception as e:
                        pass # Ignora errori download immagine, procede solo col testo

            # Pulizia del DOM per ottimizzare il contesto testuale
            for s in soup(['script', 'style', 'nav', 'footer']): s.decompose()
            raw_text = soup.get_text(separator=' ', strip=True)[:10000]
        except Exception:
            return "error"

    # Phase 2: AI Interpretation (Multimodale)
    try:
        # Costruzione dinamica del messaggio utente
        user_message = {
            'role': 'user',
            'content': ANALYSIS_PROMPT_TEMPLATE.format(ticker=ticker, title=title, raw_text=raw_text)
        }
        
        # Se abbiamo catturato un'immagine, la passiamo al modello Vision
        if image_bytes:
            user_message['images'] = [image_bytes]

        response = chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
                user_message
            ],
            options={'temperature': 0.1, 'num_ctx': 8192}
        )
        
        content_out = response.get('message', {}).get('content')
        if content_out is None:
            return "error"

        # Gatekeeper Semantico: scarta se il modello ha identificato la news come irrilevante
        if "SKIP" in content_out[:10].upper():
            return "skipped"

        # Phase 3: Storage (Naming Convention TIER C)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(title.encode('utf-8', errors='ignore')).hexdigest()[:6]
        filename = f"NEWS_{ticker}_{timestamp}_{file_hash}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content_out)
        
        # (Opzionale) Puoi usare tqdm.write anche qui per confermare il salvataggio senza rompere la barra
        # tqdm.write(f"   ✅ Salvato: {filename}")
        
        return "saved"
    except Exception as e:
        return "error"

def run_scraping_session(tickers):
    """
    Orchestra la sessione di scraping iterando sulla lista dei ticker target.
    Utilizza barre di avanzamento nidificate (tqdm) per un feedback visivo in console.
    
    Args:
        tickers (list): Lista di stringhe rappresentanti i ticker (es. ["^GSPC", "BTC-USD"]).
    """
    ticker_pbar = tqdm(tickers, desc="📈 Market Focus", unit="ticker")
    for ticker in ticker_pbar:
        ticker_pbar.set_postfix({"current": ticker})
        news_list = get_ticker_news(ticker)
        
        if not news_list: 
            continue
            
        news_pbar = tqdm(news_list, desc=f"  └─ {ticker}", unit="news", leave=False)
        for item in news_pbar:
            result = agent_process_and_save(item, ticker)
            news_pbar.set_postfix({"status": result})

if __name__ == "__main__":
    print(f"🚀 Fin-Studio Scraper: Targeting {MODEL_NAME} (Multimodal Mode)")
    # Indicatori core del Modulo 1 (Mercati, Volatilità, Valute, Crypto, Commodities)
    market_focus = ["^GSPC", "^VIX", "EURUSD=X", "BTC-USD", "GC=F", "CL=F"] 
    run_scraping_session(market_focus)
    print("\n✨ Scraping session completed. Files are ready for TIER_C Ingestion.")