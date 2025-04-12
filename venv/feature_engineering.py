import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Create features from raw data including percentage changes, 
    rolling statistics, and other technical indicators
    """
    # Percentage changes
    df['flow_mean_pct_change'] = df['flow_mean'].pct_change()
    df['flow_total_pct_change'] = df['flow_total'].pct_change()

    # Rolling statistics (6-hour window)
    df['flow_mean_ma_6'] = df['flow_mean'].rolling(window=6).mean()
    df['flow_total_ma_6'] = df['flow_total'].rolling(window=6).mean()
    df['flow_mean_std_6'] = df['flow_mean'].rolling(window=6).std()
    df['flow_total_std_6'] = df['flow_total'].rolling(window=6).std()

    # Transaction-based features
    df['tx_count_diff'] = df['transactions_count_flow'].diff()
    df['tx_count_rolling_mean'] = df['transactions_count_flow'].rolling(window=6).mean()

    # Handle NaN values and infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def normalize_features(df):
    """
    Normalize features using StandardScaler
    """
    features_to_scale = [
        'flow_mean', 'flow_total', 'transactions_count_flow',
        'flow_mean_pct_change', 'flow_total_pct_change',
        'flow_mean_ma_6', 'flow_total_ma_6',
        'flow_mean_std_6', 'flow_total_std_6',
        'tx_count_diff', 'tx_count_rolling_mean'
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df, features_to_scale

def create_labels(df, future_periods=3, threshold=0.05):
    """
    Create trading signals based on future price movements
    """
    # Shift forward to look at the future
    future_flow = df['flow_total'].shift(-future_periods)
    flow_now = df['flow_total']

    # Create labels: -1 (Sell), 0 (Hold), 1 (Buy)
    df['label'] = 0  # Default: Hold
    df.loc[(future_flow - flow_now) / flow_now > threshold, 'label'] = 1    # Buy
    df.loc[(future_flow - flow_now) / flow_now < -threshold, 'label'] = -1  # Sell
    
    return df

def prepare_sequences(df, features, sequence_length=12):
    """
    Create sequences for time series model input
    """
    X = []
    y = []

    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length][features].values)
        y.append(df.iloc[i+sequence_length]['label'])

    X = np.array(X)
    y = np.array(y)
    
    return X, y

if __name__ == "__main__":
    # Test feature engineering independently
    import asyncio
    from data_loader import fetch_data
    
    async def test_features():
        df = await fetch_data()
        df = engineer_features(df)
        df, features = normalize_features(df)
        df = create_labels(df)
        X, y = prepare_sequences(df, features)
        
        print("Feature DataFrame Shape:", df.shape)
        print("Feature Columns:", df.columns.tolist())
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Label distribution:", np.unique(y, return_counts=True))
    
    asyncio.run(test_features())