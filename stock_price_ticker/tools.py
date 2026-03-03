import yfinance as yf
from typing import List, Dict, Optional

def get_options_strike_prices(ticker: str, expiration_date: Optional[str] = None) -> Dict:
    """
    Get the currently trading options strike prices for a given stock ticker from Yahoo Finance.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'TSLA', 'SPY')
        expiration_date: Optional specific expiration date in YYYY-MM-DD format. 
                        If not provided, uses the nearest expiration date.
    
    Returns:
        Dictionary containing calls and puts strike prices, expiration date used, and underlying price.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations:
            return {"error": f"No options data available for ticker {ticker}"}
        
        # Use provided expiration or nearest available
        target_date = expiration_date if expiration_date else expirations[0]
        
        if target_date not in expirations:
            return {
                "error": f"Expiration date {target_date} not available", 
                "available_expirations": list(expirations)
            }
        
        # Get options chain for the selected expiration
        opt_chain = stock.option_chain(target_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Get underlying price
        try:
            underlying_price = stock.info.get('regularMarketPrice') or stock.fast_info.get('lastPrice')
        except:
            underlying_price = None
        
        return {
            "ticker": ticker.upper(),
            "expiration_date": target_date,
            "underlying_price": underlying_price,
            "available_expirations": list(expirations),
            "calls": {
                "strike_prices": sorted(calls['strike'].tolist()),
                "count": len(calls)
            },
            "puts": {
                "strike_prices": sorted(puts['strike'].tolist()),
                "count": len(puts)
            }
        }
    except Exception as e:
        return {"error": f"Error fetching options data: {str(e)}"}

def get_available_expirations(ticker: str) -> List[str]:
    """
    Get list of available options expiration dates for a given stock ticker.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        List of expiration dates in YYYY-MM-DD format
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        return list(expirations) if expirations else []
    except Exception as e:
        return [f"Error fetching expirations: {str(e)}"]
