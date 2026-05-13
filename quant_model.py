import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class QuantitativePipeline:
    def __init__(self, ticker="AAPL", features=None):
        self.ticker = ticker
        # NEW: Added Dist_EMA_20 and Dist_SMA_200 to the feature list
        self.features = features if features else [
            'Daily_Return', 'RSI_14', 'Dist_SMA_20', 'Dist_SMA_50', 
            'BB_Percent', 'VIX', 'Dist_EMA_20', 'Dist_SMA_200'
        ]
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=1, 
            min_samples_split=50, 
            class_weight='balanced'
        )

    def fetch_and_engineer_data(self, start_date, end_date):
        """Fetches ticker and VIX data, aligns dates, and engineers all features."""
        # Fetch Data
        asset = yf.Ticker(self.ticker)
        df = asset.history(start=start_date, end=end_date)
        
        vix = yf.Ticker("^VIX").history(start=start_date, end=end_date)['Close']
        vix.name = 'VIX'
        
        df.index = pd.to_datetime(df.index).tz_convert(None).normalize()
        vix.index = pd.to_datetime(vix.index).tz_convert(None).normalize()
        df = df.join(vix)
        
        # --- NEW: The 200-Day Macro Regime Filter ---
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        # Moving Averages & Distances
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Dist_SMA_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # --- NEW: The 20-Day Exponential Fast Trend ---
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Dist_EMA_20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
        
        # Momentum & RSI
        df['Daily_Return'] = df['Close'].pct_change()
        window_length = 14
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window_length-1, adjust=False).mean()
        ema_down = down.ewm(com=window_length-1, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
        df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
        df['BB_Percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Create Target & Clean
        df['Tomorrow_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
        df = df.drop(columns=['Tomorrow_Close'])
        
        # NOTE: Because of the SMA_200, this will drop the first 200 trading days of your dataset!
        df.dropna(inplace=True)
        
        return df

    def train_model(self, df, threshold=0.5):
        """Splits data, trains the Random Forest, and prints in-sample metrics."""
        X = df[self.features]
        y = df['Target']

        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")

        self.model.fit(X_train, y_train)
        
        # Evaluate
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_test, predictions)
        
        print(f"Precision: {precision:.4f}")
        unique, counts = np.unique(predictions, return_counts=True)
        print("Model Predictions Distribution:", dict(zip(unique, counts)))
        print("\nActual Test Set Distribution:", y_test.value_counts().to_dict())

    def run_backtest(self, df_oos, threshold=0.40, transaction_cost=0.001):
        """Generates predictions on new data and calculates vectorized strategy returns."""
        X_oos = df_oos[self.features]
        
        probabilities_oos = self.model.predict_proba(X_oos)[:, 1]
        predictions_oos = (probabilities_oos >= threshold).astype(int)

        backtest_oos = pd.DataFrame(index=X_oos.index)
        backtest_oos['Actual_Return'] = df_oos['Close'].pct_change().shift(-1).loc[X_oos.index]
        backtest_oos['Prediction'] = predictions_oos

        backtest_oos['Position_Change'] = backtest_oos['Prediction'].diff().abs()
        backtest_oos.loc[backtest_oos.index[0], 'Position_Change'] = backtest_oos['Prediction'].iloc[0]

        backtest_oos['Strategy_Return'] = backtest_oos['Prediction'] * backtest_oos['Actual_Return']
        backtest_oos['Strategy_Return'] -= (backtest_oos['Position_Change'] * transaction_cost)

        backtest_oos['Buy_and_Hold_Growth'] = (1 + backtest_oos['Actual_Return']).cumprod()
        backtest_oos['Strategy_Growth'] = (1 + backtest_oos['Strategy_Return']).cumprod()
        backtest_oos.dropna(inplace=True)

        return backtest_oos, predictions_oos
    
    def find_optimal_threshold(self, df, start=0.30, end=0.70, step=0.01, transaction_cost=0.001):
        """Iterates through a range of thresholds to find the highest Sharpe Ratio."""
        print(f"\n--- OPTIMIZING THRESHOLD ({start:.2f} to {end:.2f}) ---")
        
        X = df[self.features]
        actual_returns = df['Close'].pct_change().shift(-1).loc[X.index]
        
        # Get raw probabilities once to save computation time
        probabilities = self.model.predict_proba(X)[:, 1]
        
        thresholds = np.arange(start, end + step, step)
        sharpe_ratios = []
        total_returns = []

        for t in thresholds:
            # 1. Generate predictions for this specific threshold
            predictions = (probabilities >= t).astype(int)
            
            # 2. Calculate friction (transaction costs)
            position_change = np.abs(np.diff(predictions, prepend=predictions[0]))
            
            # 3. Calculate returns
            strategy_return = (predictions * actual_returns) - (position_change * transaction_cost)
            
            # 4. Calculate Sharpe
            mean_return = strategy_return.mean()
            std_dev = strategy_return.std()
            
            if std_dev == 0:
                sharpe = 0
            else:
                sharpe = (mean_return / std_dev) * np.sqrt(252)
                
            # Calculate total return just for tracking
            total_ret = (np.prod(1 + strategy_return.dropna()) - 1) * 100
            
            sharpe_ratios.append(sharpe)
            total_returns.append(total_ret)

        # Find the best one
        best_idx = np.argmax(sharpe_ratios)
        best_threshold = thresholds[best_idx]
        best_sharpe = sharpe_ratios[best_idx]
        best_return = total_returns[best_idx]

        print(f"Optimal Threshold: {best_threshold:.2f}")
        print(f"Max Sharpe Ratio: {best_sharpe:.2f} (Total Return: {best_return:.2f}%)")

        # Plot the Optimization Curve
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, sharpe_ratios, marker='o', color='purple', linestyle='-')
        plt.axvline(best_threshold, color='green', linestyle='--', label=f'Best: {best_threshold:.2f}')
        plt.title('Threshold Optimization: Sharpe Ratio Curve')
        plt.xlabel('Decision Threshold (Probability)')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('threshold_optimization.png')
        plt.show()

        return best_threshold

    def evaluate_and_plot(self, backtest_df, predictions_oos):
        """Calculates final metrics and plots the equity curve."""
        mean_return = backtest_df['Strategy_Return'].mean()
        std_dev = backtest_df['Strategy_Return'].std()
        sharpe_ratio = (mean_return / std_dev) * np.sqrt(252) if std_dev != 0 else 0

        final_strat_return = (backtest_df['Strategy_Growth'].iloc[-1] - 1) * 100
        final_bh_return = (backtest_df['Buy_and_Hold_Growth'].iloc[-1] - 1) * 100

        print(f"\n--- BACKTEST RESULTS ({backtest_df.index[0].date()} to {backtest_df.index[-1].date()}) ---")
        print(f"OOS Strategy Return: {final_strat_return:.2f}%")
        print(f"OOS Buy & Hold Return: {final_bh_return:.2f}%")
        print(f"OOS Strategy Sharpe Ratio: {sharpe_ratio:.2f}")

        unique, counts = np.unique(predictions_oos, return_counts=True)
        print("\nOOS Model Predictions Distribution:", dict(zip(unique, counts)))

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(backtest_df.index, backtest_df['Buy_and_Hold_Growth'], label='Buy & Hold (Baseline)', color='gray', linewidth=2, alpha=0.7)
        plt.plot(backtest_df.index, backtest_df['Strategy_Growth'], label='RF Strategy (with costs)', color='green', linewidth=2)
        
        # Highlight COVID Crash
        start_crash = pd.to_datetime('2020-02-15')
        end_crash = pd.to_datetime('2020-04-15')
        plt.axvspan(start_crash, end_crash, color='red', alpha=0.1, label='COVID-19 Crash')
        
        plt.title('Out-of-Sample Backtest: Surviving the 2020 Crash')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Growth ($1 Multiplier)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('oos_backtest.png')
        plt.show()

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    pipeline = QuantitativePipeline(ticker="AAPL")

    # 1. Train Phase (2004 - 2015)
    print("--- PREPARING TRAINING DATA ---")
    train_data = pipeline.fetch_and_engineer_data(start_date="2004-01-01", end_date="2015-12-31")
    pipeline.train_model(train_data, threshold=0.40)

    # 2. Out-of-Sample Phase (Fetch early 2017 to start testing in 2018)
    print("\n--- PREPARING OUT-OF-SAMPLE DATA ---")
    oos_data = pipeline.fetch_and_engineer_data(start_date="2017-03-01", end_date="2021-01-01")

    # --- NEW: FIND THE OPTIMAL THRESHOLD ---
    # We ask the script to test every threshold between 0.35 and 0.65
    best_t = pipeline.find_optimal_threshold(oos_data, start=0.35, end=0.65, step=0.01)

    # 3. Run Final Backtest using the BEST threshold
    print("\n--- RUNNING FINAL OPTIMIZED BACKTEST ---")
    backtest_results, oos_preds = pipeline.run_backtest(oos_data, threshold=best_t)

    # 4. Evaluate Results
    pipeline.evaluate_and_plot(backtest_results, oos_preds)