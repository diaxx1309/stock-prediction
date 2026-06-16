"""
Stock Price Prediction - Training Pipeline
============================================
Complete pipeline:
1. Load S&P 500 data
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Train-Test Split
5. Train 3 models
6. Evaluate & Compare
7. Save best model
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from model import StockPriceModels
import warnings
warnings.filterwarnings('ignore')


class StockPricePipeline:
    """Complete training pipeline for stock price prediction"""
    
    def __init__(self, data_path='data/spx.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = StockPriceModels()
        self.results = {}
        
    
    # ===== STEP 1: LOAD DATA =====
    def load_data(self):
        """Load S&P 500 data from CSV"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"   Data shape: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")
        print(f"   Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        return self.df
    
    
    # ===== STEP 2: EXPLORATORY DATA ANALYSIS =====
    def explore_data(self):
        """Understand the data structure and patterns"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        # Check for missing values
        print("\n📊 Missing Values:")
        missing = self.df.isnull().sum()
        print(f"   {missing.sum()} total missing values")
        if missing.sum() > 0:
            print(f"   Columns with missing: {missing[missing > 0]}")
        
        # Basic statistics
        print("\n📈 Price Statistics:")
        price_stats = self.df['Close'].describe()
        print(f"   Mean:   ${price_stats['mean']:.2f}")
        print(f"   Min:    ${price_stats['min']:.2f}")
        print(f"   Max:    ${price_stats['max']:.2f}")
        print(f"   StdDev: ${price_stats['std']:.2f}")
        
        # Data types
        print("\n🔍 Data Types:")
        print(self.df.dtypes)
        
        return self.df.describe()
    
    
    # ===== STEP 3: FEATURE ENGINEERING =====
    def create_features(self):
        """
        Create features from raw data
        
        FEATURE ENGINEERING EXPLAINED:
        Features are inputs to our ML models. Good features help models learn better.
        
        Features we'll create:
        1. Moving Averages (MA): smoothed price trends
           - MA_5 = average price last 5 days
           - MA_20 = average price last 20 days
           - Why: stock prices follow trends, moving avg captures this
        
        2. Lag Features: previous day prices
           - Price_lag_1 = yesterday's price
           - Price_lag_5 = price 5 days ago
           - Why: stock prices depend on recent history
        
        3. Daily Returns: percentage change
           - Return = (today - yesterday) / yesterday
           - Why: captures volatility and momentum
        
        4. Volume features: trading activity
           - Why: high volume = strong momentum
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df = self.df.copy()
        
        # Sort by date to ensure chronological order
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # === MOVING AVERAGES ===
        # 5-day moving average: average of last 5 days
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        # 20-day moving average: longer-term trend
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        # 50-day moving average: even longer-term
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # === LAG FEATURES ===
        # Previous day's price
        df['Price_lag_1'] = df['Close'].shift(1)
        # 5 days ago
        df['Price_lag_5'] = df['Close'].shift(5)
        # 10 days ago
        df['Price_lag_10'] = df['Close'].shift(10)
        
        # === DAILY RETURNS ===
        # Percentage change from previous day
        df['Daily_Return'] = df['Close'].pct_change()
        
        # === VOLUME FEATURES ===
        # Volume scaled (normalized)
        df['Volume_norm'] = df['Volume'] / df['Volume'].mean()
        
        # Remove rows with NaN (from moving average and lag features)
        df = df.dropna()
        
        print(f"✅ Features created!")
        print(f"   New columns: {[col for col in df.columns if col not in self.df.columns]}")
        print(f"   Rows after removing NaN: {len(df)}")
        
        self.df = df
        return df
    
    
    # ===== STEP 4: TRAIN-TEST SPLIT =====
    def prepare_train_test(self, test_size=0.2):
        """
        Split data into training and testing sets
        
        WHY SPLIT?
        - Train: teach model on 80% of historical data
        - Test: evaluate on unseen 20% (future) data
        - Prevents overfitting: model shouldn't memorize data
        
        TIME-SERIES AWARE SPLIT:
        - Important: don't shuffle! Keep chronological order
        - Model learns past to predict future
        - Test data = most recent data (simulates real-world prediction)
        """
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT")
        print("="*60)
        
        # Select features (everything except Date and Close)
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'Close']]
        X = self.df[feature_cols].values
        y = self.df['Close'].values
        
        # Calculate split point (80-20 split)
        split_idx = int(len(X) * (1 - test_size))
        
        # TIME-SERIES SPLIT (don't shuffle!)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # === SCALING: NORMALIZE DATA ===
        # Why scale? ML models learn better on 0-1 range
        # MinMax Scaler: transforms values to 0-1 range
        # Formula: (x - min) / (max - min)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"✅ Train-Test Split Complete!")
        print(f"   Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
        print(f"   Test set: {len(X_test)} samples ({test_size*100:.0f}%)")
        print(f"   Features: {X_train_scaled.shape[1]}")
        print(f"   Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    
    # ===== STEP 5: TRAIN MODELS =====
    def train_all_models(self):
        """Train all three models on prepared data"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # --- LINEAR REGRESSION ---
        self.models.build_linear_regression()
        self.models.train_linear_regression(self.X_train, self.y_train)
        y_pred_lr = self.models.predict_linear_regression(self.X_test)
        self.results['Linear Regression'] = self.models.evaluate_model(
            self.y_test, y_pred_lr, 'Linear Regression'
        )
        
        # --- RANDOM FOREST ---
        self.models.build_random_forest()
        self.models.train_random_forest(self.X_train, self.y_train)
        y_pred_rf = self.models.predict_random_forest(self.X_test)
        self.results['Random Forest'] = self.models.evaluate_model(
            self.y_test, y_pred_rf, 'Random Forest'
        )
        
        # --- LSTM ---
        # LSTM expects 3D input: (samples, timesteps, features)
        # Reshape for LSTM: add sequence dimension
        X_train_lstm =self.X_train
        X_test_lstm = self.X_test
        
        self.models.build_lstm()
        self.models.train_lstm(X_train_lstm, self.y_train)
        y_pred_lstm = self.models.predict_lstm(X_test_lstm).flatten()
        self.results['LSTM'] = self.models.evaluate_model(
            self.y_test, y_pred_lstm, 'LSTM'
        )
        
        return self.results
    
    
    # ===== STEP 6: COMPARE & SAVE =====
    def compare_and_save(self):
        """Compare models and save best one"""
        print("\n" + "="*60)
        
        # Compare models
        comparison_df = self.models.compare_models(self.results)
        
        # Find best model by R² score (highest = best)
        best_model_name = comparison_df['r2'].idxmax()
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print("="*60)
        
        # Save best model and scaler
        model_dict = {
            'best_model': best_model_name,
            'linear_regression': self.models.lr_model,
            'random_forest': self.models.rf_model,
            'lstm': self.models.lstm_model,
            'scaler': self.scaler,
            'results': self.results
        }
        
        with open('saved_model.pkl', 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"✅ Models saved to 'saved_model.pkl'")
        
        return best_model_name, comparison_df


# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = StockPricePipeline('spx.csv')
    
    # Execute pipeline
    pipeline.load_data()
    pipeline.explore_data()
    pipeline.create_features()
    pipeline.prepare_train_test()
    pipeline.train_all_models()
    best_model, comparison = pipeline.compare_and_save()
    
    print("\n✨ TRAINING COMPLETE! ✨")
    print(f"Best model: {best_model}")
    print("\nNext: Run app.py to deploy the model!")