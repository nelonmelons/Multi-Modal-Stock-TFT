"""
News data fetching and embedding module.
Fetches news headlines and generates embeddings using BERT or similar models.
"""

from typing import List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')


# Global model cache to avoid reloading models multiple times in same session
_MODEL_CACHE = {}


def fetch_news_embeddings(symbols: List[str], start: str, end: str,
                          api_key: Optional[str], embedding_dim: int = 768) -> pd.DataFrame:
    """
    For each symbol/date, fetch relevant news headlines/summaries and generate embeddings.
    
    Args:
        symbols: List of stock symbols
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        api_key: News API key
        embedding_dim: Dimension of embeddings (default 768 for BERT-base)
        
    Returns:
        DataFrame with columns: symbol, date, emb_0 .. emb_{N-1}, sentiment_score
    """
    print(f"Fetching news embeddings for {len(symbols)} symbols")
    
    if not api_key:
        print("Warning: No API key provided, returning empty news data")
        return create_empty_news_df(symbols, start, end, embedding_dim)
    
    # Check if dates are within NewsAPI's free tier limitations (last 30 days)
    from datetime import datetime, timedelta
    newsapi_cutoff = datetime.now() - timedelta(days=30)
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    
    if end_date < newsapi_cutoff:
        print(f"📅 Warning: Requested dates ({start} to {end}) are older than NewsAPI free tier limit")
        print(f"   NewsAPI free tier only allows access to news from the last 30 days")
        print(f"   Returning zero embeddings for historical data compatibility")
        return create_empty_news_df(symbols, start, end, embedding_dim)
    
    if start_date < newsapi_cutoff:
        print(f"📅 Note: Start date {start} is older than NewsAPI limit, adjusting to recent dates")
        start = newsapi_cutoff.strftime('%Y-%m-%d')
        print(f"   Fetching news from {start} to {end} (recent data only)")
    
    print(f"   Using NewsAPI date range: {start} to {end}")
    
    # Get cached FinBERT model for financial text embeddings
    model_name = "yiyanghkust/finbert-tone"
    tokenizer, model = get_cached_model(model_name)
    
    if tokenizer is None or model is None:
        print("❌ Could not load embedding model, returning empty news data")
        return create_empty_news_df(symbols, start, end, embedding_dim)
    
    all_news_data = []
    
    # Generate date range (original range for DataFrame structure)
    original_start_date = datetime.strptime(start if start_date >= newsapi_cutoff else start, '%Y-%m-%d')
    original_end_date = datetime.strptime(end, '%Y-%m-%d')
    original_date_range = pd.date_range(start=original_start_date, end=original_end_date, freq='D')
    
    # Generate news-available date range (within API limits)
    news_start_date = max(start_date, newsapi_cutoff)
    news_end_date = end_date
    news_date_range = pd.date_range(start=news_start_date, end=news_end_date, freq='D')
    
    for symbol in symbols:
        print(f"  Processing news for {symbol}...")
        
        # Fetch news for this symbol (only for dates within API limits)
        news_articles = []
        if news_date_range.size > 0:
            news_articles = fetch_news_for_symbol(
                symbol, 
                news_start_date.strftime('%Y-%m-%d'), 
                news_end_date.strftime('%Y-%m-%d'), 
                api_key
            )
        
        for date in original_date_range:
            date_str = date.strftime('%Y-%m-%d')
            
            # Check if this date has news available (within API limits)
            if date >= news_start_date and news_articles:
                # Get news for this specific date
                daily_news = [
                    article for article in news_articles 
                    if article.get('date', '')[:10] == date_str
                ]
                
                # Create embedding for this symbol-date combination
                embedding, sentiment = create_news_embedding(
                    daily_news, tokenizer, model, embedding_dim
                )
            else:
                # Use zero embedding for dates outside API limits
                embedding, sentiment = np.zeros(embedding_dim), 0.0
            
            # Create row data
            row_data = {
                'symbol': symbol,
                'date': date
            }
            
            # Add embedding dimensions
            for i, emb_val in enumerate(embedding):
                row_data[f'emb_{i}'] = emb_val
            
            row_data['sentiment_score'] = sentiment
            
            all_news_data.append(row_data)
    
    if not all_news_data:
        return create_empty_news_df(symbols, start, end, embedding_dim)
    
    news_df = pd.DataFrame(all_news_data)
    print(f"Generated news embeddings: {len(news_df)} rows with {embedding_dim} dimensions")
    
    return news_df


def fetch_news_for_symbol(symbol: str, start: str, end: str, api_key: str) -> List[dict]:
    """
    Fetch news articles for a specific symbol using NewsAPI.
    
    Args:
        symbol: Stock symbol
        start: Start date
        end: End date  
        api_key: API key
        
    Returns:
        List of news article dictionaries
    """
    articles = []
    
    try:
        # Fetch from NewsAPI
        articles = fetch_from_newsapi(symbol, start, end, api_key)
        
        # Print headlines for verification
        if articles:
            print(f"    📰 Sample headlines for {symbol}:")
            for i, article in enumerate(articles[:3]):  # Show first 3 headlines
                title = article.get('title', 'No title')[:80] + ('...' if len(article.get('title', '')) > 80 else '')
                print(f"       {i+1}. {title}")
            
    except Exception as e:
        print(f"    Warning: Error fetching news for {symbol}: {e}")
    
    print(f"    Found {len(articles)} articles for {symbol}")
    return articles


def fetch_from_newsapi(symbol: str, start: str, end: str, api_key: str) -> List[dict]:
    """Fetch news from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': f'"{symbol}" OR "{get_company_name(symbol)}"',
        'from': start,
        'to': end,
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'language': 'en',
        'pageSize': 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'date': article.get('publishedAt', ''),
                    'source': 'newsapi'
                })
            return articles
        elif response.status_code == 426:
            # Historical data limitation
            print(f"    📅 NewsAPI historical data limitation: dates too far in past")
            return []
        else:
            print(f"    NewsAPI error {response.status_code}: {response.text[:100]}")
            return []
    except Exception as e:
        print(f"    NewsAPI error: {e}")
    
    return []


def fetch_from_finnhub_news(symbol: str, start: str, end: str, api_key: str) -> List[dict]:
    """
    DEPRECATED: Finnhub news support has been removed.
    Corporate actions now use yfinance exclusively.
    """
    print(f"    Warning: Finnhub news support has been removed. Use NewsAPI instead.")
    return []


def create_news_embedding(articles: List[dict], tokenizer, model, 
                         embedding_dim: int) -> tuple:
    """
    Create a single embedding vector from multiple news articles.
    
    Args:
        articles: List of news articles
        tokenizer: BERT tokenizer
        model: BERT model
        embedding_dim: Target embedding dimension
        
    Returns:
        Tuple of (embedding_vector, sentiment_score)
    """
    if not articles:
        # Return zero embedding for no news
        return np.zeros(embedding_dim), 0.0
    
    # Combine all article text
    combined_text = ""
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        combined_text += f"{title} {description} "
    
    combined_text = combined_text.strip()
    if not combined_text:
        return np.zeros(embedding_dim), 0.0
    
    try:
        # Tokenize and encode
        inputs = tokenizer(
            combined_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Generate embedding and sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            
            if hasattr(outputs, 'logits'):
                # FinBERT classification model - extract sentiment from logits
                logits = outputs.logits
                # FinBERT outputs: [negative, neutral, positive]
                probabilities = torch.softmax(logits, dim=-1)
                sentiment_score = float(probabilities[0, 2] - probabilities[0, 0])  # positive - negative
                
                # For embedding, use the hidden states if available
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    embedding = outputs.hidden_states[-1][:, 0, :].numpy().flatten()
                else:
                    # Use the classifier's hidden representation
                    embedding = logits.numpy().flatten()
                    # Expand to desired dimension if needed
                    if len(embedding) < embedding_dim:
                        repeat_factor = embedding_dim // len(embedding) + 1
                        embedding = np.tile(embedding, repeat_factor)[:embedding_dim]
            else:
                # Standard BERT model - use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                sentiment_score = float(np.tanh(np.mean(embedding)))  # Simple sentiment from embedding
        
        # Truncate or pad to desired dimension
        if len(embedding) > embedding_dim:
            embedding = embedding[:embedding_dim]
        elif len(embedding) < embedding_dim:
            padding = np.zeros(embedding_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        return embedding, sentiment_score
        
    except Exception as e:
        print(f"    Warning: Error creating embedding: {e}")
        return np.zeros(embedding_dim), 0.0


def create_empty_news_df(symbols: List[str], start: str, end: str, 
                        embedding_dim: int) -> pd.DataFrame:
    """Create an empty news DataFrame with proper structure."""
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for symbol in symbols:
        for date in date_range:
            row = {'symbol': symbol, 'date': date}
            # Add zero embeddings
            for i in range(embedding_dim):
                row[f'emb_{i}'] = 0.0
            row['sentiment_score'] = 0.0
            data.append(row)
    
    return pd.DataFrame(data)


def get_company_name(symbol: str) -> str:
    """Get company name for better news search."""
    # Simple mapping for common symbols
    company_map = {
        'AAPL': 'Apple',
        'GOOGL': 'Google Alphabet',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'NVDA': 'Nvidia'
    }
    return company_map.get(symbol, symbol)


def get_cached_model(model_name: str = "yiyanghkust/finbert-tone"):
    """
    Get a cached BERT model and tokenizer.
    
    Models are cached both:
    1. On disk by Hugging Face (~/.cache/huggingface/transformers)
    2. In memory during the session to avoid reloading
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (tokenizer, model) or (None, None) if failed
    """
    if model_name in _MODEL_CACHE:
        print(f"🚀 Using cached model: {model_name}")
        return _MODEL_CACHE[model_name]
    
    try:
        print(f"📦 Loading FinBERT model: {model_name}")
        print("   FinBERT: Financial domain-specific BERT for sentiment analysis")
        print("   First time: downloading and caching (~400MB)")
        print("   Future uses: loading from cache (much faster)")
        
        if "finbert" in model_name.lower():
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        
        model.eval()
        
        # Cache in memory for this session
        _MODEL_CACHE[model_name] = (tokenizer, model)
        
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        print(f"✅ Model loaded and cached at: {cache_dir}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None, None


def preload_embedding_model(model_name: str = "yiyanghkust/finbert-tone"):
    """
    Pre-download and cache the embedding model.
    
    This function can be called once to ensure the model is cached
    before running the main pipeline.
    
    Args:
        model_name: Name of the model to preload
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🔄 Pre-loading embedding model: {model_name}")
    tokenizer, model = get_cached_model(model_name)
    
    if tokenizer is not None and model is not None:
        print(f"✅ Model {model_name} successfully cached!")
        return True
    else:
        print(f"❌ Failed to cache model {model_name}")
        return False


def check_model_cache_size():
    """Check the size of the Hugging Face model cache."""
    import os
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    
    if not os.path.exists(cache_dir):
        print(f"📁 Cache directory not found: {cache_dir}")
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass
    
    size_mb = total_size / (1024 * 1024)
    print(f"📊 Model cache size: {size_mb:.1f} MB")
    print(f"📁 Cache location: {cache_dir}")
    
    return size_mb


def clear_model_cache():
    """Clear the in-memory model cache (not the disk cache)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print("🧹 Cleared in-memory model cache")


def check_newsapi_date_limits(api_key: str) -> dict:
    """
    Check NewsAPI date limitations and account status.
    
    Args:
        api_key: NewsAPI key
        
    Returns:
        Dictionary with limitation info
    """
    from datetime import datetime, timedelta
    
    result = {
        'api_working': False,
        'free_tier_limit_days': 30,
        'earliest_date': None,
        'current_date': datetime.now().strftime('%Y-%m-%d'),
        'recommendations': []
    }
    
    try:
        # Test API status
        url = 'https://newsapi.org/v2/sources'
        response = requests.get(url, params={'apiKey': api_key}, timeout=5)
        
        if response.status_code == 200:
            result['api_working'] = True
            
            # Calculate earliest accessible date (approximately 30 days back for free tier)
            earliest = datetime.now() - timedelta(days=30)
            result['earliest_date'] = earliest.strftime('%Y-%m-%d')
            
            # Test historical access to get exact limitation
            test_url = 'https://newsapi.org/v2/everything'
            old_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            test_response = requests.get(test_url, params={
                'q': 'test',
                'from': old_date,
                'to': old_date,
                'apiKey': api_key
            }, timeout=5)
            
            if test_response.status_code == 426:
                # Parse the exact cutoff from error message
                try:
                    error_data = test_response.json()
                    message = error_data.get('message', '')
                    if 'as far back as' in message:
                        # Extract date from message like "as far back as 2025-05-18"
                        import re
                        date_match = re.search(r'as far back as (\d{4}-\d{2}-\d{2})', message)
                        if date_match:
                            result['earliest_date'] = date_match.group(1)
                except:
                    pass
        
        # Add recommendations based on status
        if result['api_working']:
            result['recommendations'] = [
                f"✅ NewsAPI is working for recent dates (since {result['earliest_date']})",
                "📅 For historical data beyond 30 days, consider:",
                "   • Upgrading to NewsAPI paid plan ($449/month for historical access)",
                "   • Using alternative sources (Guardian API, Reddit API for sentiment)",
                "   • Pre-computing news embeddings for recent data and using cached values",
                "   • Using zero embeddings for historical periods (current approach)"
            ]
        else:
            result['recommendations'] = [
                "❌ NewsAPI not accessible - check API key",
                "🔧 Alternative news sources to consider:",
                "   • Guardian API (free, limited historical)",
                "   • Reddit API (free, good for sentiment)",
                "   • Alpha Vantage News (limited free tier)",
                "   • Manual news dataset (e.g., Kaggle financial news)"
            ]
            
    except Exception as e:
        result['recommendations'] = [f"❌ Error checking NewsAPI: {e}"]
    
    return result


def print_news_status_report(api_key: Optional[str] = None):
    """Print a comprehensive report on news data availability."""
    print("\n" + "=" * 60)
    print("📰 NEWS DATA AVAILABILITY REPORT")
    print("=" * 60)
    
    if not api_key:
        print("❌ No NewsAPI key provided")
        print("💡 Set NEWS_API_KEY in .env file to enable news embeddings")
        return
    
    status = check_newsapi_date_limits(api_key)
    
    print(f"🔑 API Status: {'✅ Working' if status['api_working'] else '❌ Not Working'}")
    print(f"📅 Current Date: {status['current_date']}")
    print(f"📅 Earliest Available: {status['earliest_date'] or 'Unknown'}")
    print(f"⏰ Free Tier Limit: ~{status['free_tier_limit_days']} days")
    
    print("\n📋 RECOMMENDATIONS:")
    for rec in status['recommendations']:
        print(f"   {rec}")
    
    print("\n🛠️  CURRENT PIPELINE BEHAVIOR:")
    print("   • Recent dates (< 30 days): Fetch real news embeddings")
    print("   • Historical dates (> 30 days): Use zero embeddings")
    print("   • TFT training: Works with mixed real/zero embeddings")
    print("   • Feature consistency: All news features always present")
    
    print("=" * 60)
