import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# Import custom modules
from data_loader import fetch_data
from feature_engineering import engineer_features, normalize_features, create_labels, prepare_sequences
from model import train_model, plot_confusion_matrix, plot_training_history, save_model
from backtester import Backtester

async def main():
    """
    Main function to run the complete pipeline:
    1. Load data
    2. Engineer features
    3. Train model
    4. Run backtest
    """
    print("Step 1: Loading data...")
    df = await fetch_data(
        start_date=datetime(year=2024, month=1, day=1, tzinfo=timezone.utc),
        end_date=datetime(year=2025, month=1, day=1, tzinfo=timezone.utc)
    )
    print(f"Loaded {len(df)} data points")
    
    print("\nStep 2: Engineering features...")
    df = engineer_features(df)
    df, features_list = normalize_features(df)
    df = create_labels(df, future_periods=3, threshold=0.05)
    X, y = prepare_sequences(df, features_list, sequence_length=12)
    print(f"Created {len(X)} sequences with {X.shape[1]} timesteps and {X.shape[2]} features")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    print("\nStep 3: Training model...")
    model, history, X_test, y_test, y_pred_classes, y_true_classes = train_model(
        X, y, epochs=20, batch_size=32, validation_split=0.1
    )
    
    # Plot training results
    plot_confusion_matrix(y_true_classes, y_pred_classes)
    plot_training_history(history)
    
    # Save the model for future use
    save_model(model, 'crypto_flow_model.h5')
    
    print("\nStep 4: Running backtest...")
    # Extract test period prices and dates for backtesting
    test_prices = df['flow_total'].iloc[-len(y_pred_classes):].values
    test_dates = df.index[-len(y_pred_classes):]
    
    # Initialize and run backtester
    backtester = Backtester(initial_balance=1000, fee_rate=0.0006)
    results = backtester.run_backtest(y_pred_classes, test_prices, test_dates)
    
    # Show backtest results
    backtester.print_results(results)
    backtester.plot_portfolio_performance(results)
    backtester.plot_drawdown(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    asyncio.run(main())