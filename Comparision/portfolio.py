import numpy as np
import pandas as pd

def simulate_portfolio(predictions, actuals, initial_capital=100000, top_k=5):
    """
    Simulates a portfolio strategy based on model predictions.
    
    Args:
        predictions (pd.DataFrame): DataFrame with columns ['date', 'symbol', 'prediction']
        actuals (pd.DataFrame): DataFrame with columns ['date', 'symbol', 'actual_return']
        initial_capital (float): Starting capital for the simulation.
        top_k (int): Number of top stocks to invest in based on predictions.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio's daily value and returns.
    """
    
    # Merge predictions with actual returns
    # Ensure dates are in the same format
    predictions['date'] = pd.to_datetime(predictions['date']).dt.date
    actuals['date'] = pd.to_datetime(actuals['date']).dt.date
    
    data = pd.merge(predictions, actuals, on=['date', 'symbol'], how='inner')
    
    if data.empty:
        print("Warning: No matching data between predictions and actuals for portfolio simulation.")
        return pd.DataFrame({'date': [], 'portfolio_value': [], 'daily_return': []})

    # Get unique dates from the test set
    dates = sorted(data['date'].unique())
    
    portfolio_values = []
    capital = initial_capital
    
    for date in dates:
        daily_data = data[data['date'] == date]
        
        # Select top_k stocks to invest in for the next period
        long_portfolio = daily_data.nlargest(top_k, 'prediction')['symbol'].tolist()
        
        # Assume equal weighting for simplicity
        investment_per_stock = capital / top_k if long_portfolio else 0
        
        # Calculate returns for the selected stocks
        daily_return_value = 0
        if long_portfolio:
            # The return is the average of the actual returns of the stocks we decided to hold
            portfolio_return = daily_data[daily_data['symbol'].isin(long_portfolio)]['actual_return'].mean()
            daily_return_value = portfolio_return if not np.isnan(portfolio_return) else 0

        capital *= (1 + daily_return_value)
        portfolio_values.append({'date': date, 'portfolio_value': capital})
        
    if not portfolio_values:
        return pd.DataFrame({'date': [], 'portfolio_value': [], 'daily_return': []})

    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change().fillna(0)
    
    return portfolio_df

def calculate_performance_metrics(portfolio_df):
    """
    Calculates performance metrics for a portfolio.
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with 'daily_return' column.

    Returns:
        dict: A dictionary of performance metrics.
    """
    
    if portfolio_df.empty or 'daily_return' not in portfolio_df.columns:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
        }

    # Total Return
    total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
    
    # Annualized Return
    num_days = len(portfolio_df)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0
    
    # Annualized Volatility
    annualized_volatility = portfolio_df['daily_return'].std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Max Drawdown
    cumulative_returns = (1 + portfolio_df['daily_return']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }
