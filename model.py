import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class StockPriceModels:
    
    def __init__(self):
        self.lr_model = None
        self.rf_model = None
        self.lstm_model = None
        self.scaler = None
        
    def build_linear_regression(self):
        self.lr_model = LinearRegression()
        return self.lr_model
    
    def train_linear_regression(self, X_train, y_train):
        print("🔵 Training Linear Regression...")
        self.lr_model.fit(X_train, y_train)
        return self.lr_model
    
    def predict_linear_regression(self, X_test):
        return self.lr_model.predict(X_test)
    
    def build_random_forest(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        return self.rf_model
    
    def train_random_forest(self, X_train, y_train):
        print("🟢 Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        return self.rf_model
    
    def predict_random_forest(self, X_test):
        return self.rf_model.predict(X_test)
    
    def build_lstm(self, input_shape=None):
        self.lstm_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        return self.lstm_model
    
    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        print("🔴 Training Gradient Boosting...")
        self.lstm_model.fit(X_train, y_train)
        return self.lstm_model, None
    
    def predict_lstm(self, X_test):
        return self.lstm_model.predict(X_test).reshape(-1, 1)
    
    @staticmethod
    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📊 {model_name} Results:")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   R²:   {r2:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    @staticmethod
    def compare_models(results_dict):
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        df = pd.DataFrame(results_dict).T
        df = df.round(4)
        print(df)
        
        best_mae = df['mae'].idxmin()
        best_rmse = df['rmse'].idxmin()
        best_r2 = df['r2'].idxmax()
        
        print(f"\n🏆 Best Model by Metric:")
        print(f"   MAE:  {best_mae}")
        print(f"   RMSE: {best_rmse}")
        print(f"   R²:   {best_r2}")
        
        return df
