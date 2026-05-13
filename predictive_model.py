
---

## **FILE 2: `predictive_model.py`**

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📈 PREDICTIVE ANALYTICS PROJECT")
print("="*60)

# ============================================
# 1. GENERATE HISTORICAL DATA
# ============================================
print("\n📁 Step 1: Generating historical sales data...")

np.random.seed(42)
dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
n = len(dates)

# Create realistic sales pattern with trend, seasonality, and noise
trend = np.linspace(10000, 25000, n)  # Increasing trend
seasonality = 5000 * np.sin(2 * np.pi * np.arange(n) / 365)  # Yearly cycle
weekly_pattern = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly cycle
noise = np.random.normal(0, 2000, n)  # Random variation

# Special events (holiday spikes)
holiday_effect = np.zeros(n)
holiday_dates = ['2021-12-25', '2022-12-25', '2023-12-25', '2024-12-25',
                 '2021-11-26', '2022-11-25', '2023-11-24', '2024-11-29']  # Black Friday
for hd in holiday_dates:
    idx = dates.get_loc(pd.Timestamp(hd)) if hd in dates else None
    if idx:
        holiday_effect[idx:idx+7] = 8000  # Week-long boost

sales = trend + seasonality + weekly_pattern + noise + holiday_effect
sales = np.maximum(sales, 500)  # Minimum sales floor

df = pd.DataFrame({
    'date': dates,
    'sales': sales.round(2),
    'day_of_week': dates.dayofweek,
    'month': dates.month,
    'year': dates.year,
    'quarter': dates.quarter,
    'is_weekend': (dates.dayofweek >= 5).astype(int)
})

# Add promotional periods
df['promotion'] = 0
for year in [2021, 2022, 2023, 2024]:
    promo_start = pd.Timestamp(f'{year}-06-01')
    promo_end = pd.Timestamp(f'{year}-06-30')
    df.loc[(df['date'] >= promo_start) & (df['date'] <= promo_end), 'promotion'] = 1

df.to_csv('historical_sales.csv', index=False)
print(f"✅ Generated {len(df)} days of sales data (2021-2024)")

# ============================================
# 2. FEATURE ENGINEERING
# ============================================
print("\n🔧 Step 2: Engineering features for modeling...")

# Create lag features (previous days sales)
for lag in [1, 7, 30]:
    df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

# Create rolling averages
for window in [7, 30, 90]:
    df[f'rolling_mean_{window}'] = df['sales'].rolling(window).mean()
    df[f'rolling_std_{window}'] = df['sales'].rolling(window).std()

# Drop NaN values from lag/rolling features
df_clean = df.dropna().reset_index(drop=True)

# ============================================
# 3. TRAIN MODELS
# ============================================
print("\n🤖 Step 3: Training prediction models...")

# Prepare features for regression models
feature_cols = ['day_of_week', 'month', 'quarter', 'is_weekend', 'promotion',
                'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
                'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']

X = df_clean[feature_cols]
y = df_clean['sales']

# Split data (80% train, 20% test)
split_date = '2024-01-01'
train = df_clean[df_clean['date'] < split_date]
test = df_clean[df_clean['date'] >= split_date]

X_train = train[feature_cols]
y_train = train['sales']
X_test = test[feature_cols]
y_test = test['sales']

print(f"Training data: {len(train)} days (2021-2023)")
print(f"Testing data: {len(test)} days (2024)")

# Model 1: Linear Regression
print("\n📊 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Random Forest
print("🌲 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model 3: ARIMA (Time Series)
print("📈 Training ARIMA model...")
arima_data = df_clean.set_index('date')['sales']
train_arima = arima_data[arima_data.index < split_date]
test_arima = arima_data[arima_data.index >= split_date]

arima_model = ARIMA(train_arima, order=(5,1,2))  # (p,d,q) parameters
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(test_arima))

# ============================================
# 4. EVALUATE MODELS
# ============================================
print("\n📊 Step 4: Evaluating model performance...")

def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Model': model_name,
        'R² Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE %': mape
    }

results = []
results.append(evaluate_model(y_test, lr_pred, 'Linear Regression'))
results.append(evaluate_model(y_test, rf_pred, 'Random Forest'))
results.append(evaluate_model(y_test.values, arima_pred, 'ARIMA'))

results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Save results
results_df.to_csv('model_performance.csv', index=False)

# Best model
best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
best_r2 = results_df['R² Score'].max()
print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.3f})")

# ============================================
# 5. FORECAST FUTURE (NEXT 12 MONTHS)
# ============================================
print("\n🔮 Step 5: Forecasting next 12 months...")

# Use best model for forecasting
future_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
future_df = pd.DataFrame({'date': future_dates})
future_df['day_of_week'] = future_df['date'].dt.dayofweek
future_df['month'] = future_df['date'].dt.month
future_df['quarter'] = future_df['date'].dt.quarter
future_df['is_weekend'] = (future_df['date'].dt.dayofweek >= 5).astype(int)

# Add promotions (June promotion)
future_df['promotion'] = 0
future_df.loc[future_df['date'].dt.month == 6, 'promotion'] = 1

# For lag features, use last known values from historical data
last_known = df_clean.iloc[-1]
for lag in [1, 7, 30]:
    future_df[f'sales_lag_{lag}'] = last_known[f'sales_lag_{lag}']

# For rolling features, use last rolling values
for window in [7, 30]:
    future_df[f'rolling_mean_{window}'] = last_known[f'rolling_mean_{window}']
future_df['rolling_std_7'] = last_known['rolling_std_7']

# Make predictions
if best_model == 'Linear Regression':
    future_df['forecast_sales'] = lr_model.predict(future_df[feature_cols])
elif best_model == 'Random Forest':
    future_df['forecast_sales'] = rf_model.predict(future_df[feature_cols])
else:
    # Use ARIMA for forecasting
    future_arima = arima_fit.forecast(steps=365)
    future_df['forecast_sales'] = future_arima

# Add confidence intervals (based on model error)
std_error = results_df.loc[results_df['Model'] == best_model, 'RMSE'].values[0]
future_df['lower_bound'] = future_df['forecast_sales'] - 1.96 * std_error
future_df['upper_bound'] = future_df['forecast_sales'] + 1.96 * std_error
future_df['lower_bound'] = np.maximum(future_df['lower_bound'], 0)

future_df.to_csv('forecast_12months.csv', index=False)

# Monthly aggregation for easier viewing
monthly_forecast = future_df.groupby(future_df['date'].dt.strftime('%Y-%m')).agg({
    'forecast_sales': 'sum',
    'lower_bound': 'sum',
    'upper_bound': 'sum'
}).reset_index()
monthly_forecast.columns = ['month', 'forecast_sales', 'lower_bound', 'upper_bound']

print("\n📊 12-MONTH FORECAST SUMMARY:")
print("="*60)
print(monthly_forecast.to_string(index=False))
print("="*60)

# ============================================
# 6. CREATE VISUALIZATIONS
# ============================================
print("\n📊 Step 6: Creating visualizations...")

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Historical Sales Trend
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(df['date'], df['sales'], linewidth=1, color='#2E86AB', alpha=0.7, label='Actual Sales')
ax1.set_title('Historical Sales Trend (2021-2024)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales (₹)')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('historical_trend.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 2: Model Predictions vs Actual (Test Set)
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(test['date'], y_test, label='Actual', linewidth=2, color='#2E86AB')
ax2.plot(test['date'], lr_pred, label='Linear Regression', linewidth=1.5, alpha=0.7, linestyle='--')
ax2.plot(test['date'], rf_pred, label='Random Forest', linewidth=1.5, alpha=0.7, linestyle='-.')
ax2.plot(test['date'], arima_pred, label='ARIMA', linewidth=1.5, alpha=0.7, linestyle=':')
ax2.set_title('Model Predictions vs Actual (2024 Test Data)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Sales (₹)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_predictions.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 3: Model Performance Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df['Model']))
width = 0.25

bars1 = ax3.bar(x - width, results_df['R² Score'], width, label='R² Score', color='#2E86AB')
bars2 = ax3.bar(x, results_df['MAE'] / 1000, width, label='MAE (₹K)', color='#A23B72')
bars3 = ax3.bar(x + width, results_df['RMSE'] / 1000, width, label='RMSE (₹K)', color='#F18F01')

ax3.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['Model'])
ax3.set_ylabel('Score / Error (normalized)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 4: Forecast with Confidence Intervals
fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.fill_between(future_df['date'], future_df['lower_bound'], future_df['upper_bound'], 
                  alpha=0.3, color='gray', label='95% Confidence Interval')
ax4.plot(future_df['date'], future_df['forecast_sales'], linewidth=2, color='#E63946', label='Forecast')
ax4.set_title(f'{best_model} - 12 Month Sales Forecast', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Sales (₹)')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_visualization.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 5: Interactive Dashboard
fig5 = make_subplots(rows=3, cols=2, subplot_titles=('Historical Sales', 'Model Predictions (2024)',
                                                      '12-Month Forecast', 'Monthly Forecast',
                                                      'Model Comparison', 'Seasonal Pattern'))

# Historical sales
fig5.add_trace(go.Scatter(x=df['date'], y=df['sales'], mode='lines', name='Historical', 
                          line=dict(color='#2E86AB', width=1)), row=1, col=1)

# Model predictions
fig5.add_trace(go.Scatter(x=test['date'], y=y_test, mode='lines', name='Actual', 
                          line=dict(color='#2E86AB', width=2)), row=1, col=2)
fig5.add_trace(go.Scatter(x=test['date'], y=rf_pred, mode='lines', name='Random Forest', 
                          line=dict(color='#E63946', width=1.5, dash='dash')), row=1, col=2)

# Forecast with confidence
fig5.add_trace(go.Scatter(x=future_df['date'], y=future_df['forecast_sales'], mode='lines', 
                          name='Forecast', line=dict(color='#E63946', width=2)), row=2, col=1)
fig5.add_trace(go.Scatter(x=future_df['date'], y=future_df['upper_bound'], mode='lines', 
                          name='Upper Bound', line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=2, col=1)
fig5.add_trace(go.Scatter(x=future_df['date'], y=future_df['lower_bound'], mode='lines', 
                          name='Lower Bound', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', 
                          fillcolor='rgba(128,128,128,0.2)', showlegend=False), row=2, col=1)

# Monthly forecast bars
fig5.add_trace(go.Bar(x=monthly_forecast['month'], y=monthly_forecast['forecast_sales'], 
                      name='Monthly Forecast', marker_color='#F18F01'), row=2, col=2)

# Model comparison
fig5.add_trace(go.Bar(x=results_df['Model'], y=results_df['R² Score'], name='R² Score', 
                      marker_color='#2E86AB'), row=3, col=1)
fig5.add_trace(go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE', 
                      marker_color='#E63946'), row=3, col=1)

# Seasonal pattern
monthly_avg = df.groupby('month')['sales'].mean()
fig5.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers', 
                          name='Avg Monthly Sales', line=dict(color='#2E86AB', width=2)), row=3, col=2)

fig5.update_layout(title='Predictive Analytics Dashboard', height=1200, showlegend=True)
fig5.write_html('forecast_dashboard.html')
print("✅ Interactive dashboard saved: forecast_dashboard.html")

# ============================================
# 7. BUSINESS INSIGHTS
# ============================================
print("\n💡 Step 7: Generating business insights...")

# Calculate key metrics
total_2024_sales = y_test.sum()
total_2025_forecast = future_df['forecast_sales'].sum()
growth_pct = ((total_2025_forecast - total_2024_sales) / total_2024_sales) * 100

# Seasonal insights
peak_month = df.groupby('month')['sales'].mean().idxmax()
low_month = df.groupby('month')['sales'].mean().idxmin()

insights = f"""
{'='*60}
PREDICTIVE ANALYTICS INSIGHTS
{'='*60}

📊 FORECAST SUMMARY:
   • 2024 Actual Sales: ₹{total_2024_sales:,.2f}
   • 2025 Forecast Sales: ₹{total_2025_forecast:,.2f}
   • Expected Growth: {growth_pct:.1f}%

🎯 KEY FINDINGS:
   1. Best Performing Model: {best_model} (R² = {best_r2:.3f})
   2. Average Prediction Error (MAE): ₹{results_df['MAE'].mean():,.0f}
   3. Peak Sales Month: Month {peak_month} (Holiday/Summer season)
   4. Lowest Sales Month: Month {low_month} (Post-holiday slump)

📈 SEASONAL PATTERNS:
   • Q4 (Oct-Dec) shows {((df[df['quarter']==4]['sales'].mean() / df['sales'].mean())-1)*100:.0f}% higher sales
   • June promotion lifts sales by {df[df['promotion']==1]['sales'].mean() / df[df['promotion']==0]['sales'].mean():.1f}x
   • Weekend sales are {((df[df['is_weekend']==1]['sales'].mean() / df[df['is_weekend']==0]['sales'].mean())-1)*100:.0f}% lower than weekdays

🔮 FUTURE PREDICTIONS:
   • Q1 2025 (Jan-Mar): ₹{future_df[future_df['date'].dt.quarter==1]['forecast_sales'].sum():,.2f}
   • Q2 2025 (Apr-Jun): ₹{future_df[future_df['date'].dt.quarter==2]['forecast_sales'].sum():,.2f}
   • Q3 2025 (Jul-Sep): ₹{future_df[future_df['date'].dt.quarter==3]['forecast_sales'].sum():,.2f}
   • Q4 2025 (Oct-Dec): ₹{future_df[future_df['date'].dt.quarter==4]['forecast_sales'].sum():,.2f}

💡 BUSINESS RECOMMENDATIONS:

   1. INVENTORY PLANNING:
      • Increase inventory 40% in Q4 (peak season)
      • Stock up for June promotion month
      • Maintain lower inventory in {low_month}

   2. MARKETING STRATEGY:
      • Allocate 50% of Q4 budget to holiday campaigns
      • Run promotions in low months to boost sales
      • Target weekends with special offers

   3. BUDGET ALLOCATION:
      • Expected revenue: ₹{total_2025_forecast/1e6:.1f}M
      • Recommended marketing spend: ₹{total_2025_forecast * 0.1:,.0f} (10% of forecast)
      • Inventory buffer: ₹{std_error * 365:,.0f} (risk coverage)

   4. RISK MITIGATION:
      • Confidence interval range: ±₹{1.96 * std_error * 365:,.0f}
      • Keep cash reserve for downside scenarios
      • Review forecast quarterly for adjustments

{'='*60}
ACTION ITEMS
{'='*60}

✅ IMMEDIATE (Next 30 days):
   □ Prepare inventory for upcoming peak season
   □ Set up automated sales tracking
   □ Implement forecasting system

✅ SHORT-TERM (Quarterly):
   □ Review and retrain models every quarter
   □ Adjust promotions based on forecast
   □ Monitor actual vs predicted performance

✅ LONG-TERM (Annual):
   □ Build ensemble model combining all approaches
   □ Incorporate external factors (economy, competition)
   □ Automate re-forecasting pipeline

{'='*60}
"""

print(insights)

with open('forecast_insights.txt', 'w') as f:
    f.write(insights)

print("✅ Insights saved to forecast_insights.txt")

# ============================================
# 8. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 PREDICTIVE ANALYTICS PROJECT COMPLETE!")
print("="*60)
print("\n📁 Files Generated:")
print("   1. historical_sales.csv - 4 years of daily sales data")
print("   2. model_performance.csv - Accuracy metrics comparison")
print("   3. forecast_12months.csv - Daily predictions for 2025")
print("   4. forecast_dashboard.html - Interactive visualization dashboard")
print("   5. historical_trend.png - Historical sales chart")
print("   6. model_predictions.png - Predictions vs actual")
print("   7. model_performance.png - Model comparison chart")
print("   8. forecast_visualization.png - Forecast with confidence")
print("   9. forecast_insights.txt - Business insights & recommendations")
print("\n🚀 Next Steps:")
print("   1. Open forecast_dashboard.html in your browser")
print("   2. Review model performance metrics")
print("   3. Use forecast_12months.csv for planning")
print("   4. Implement recommendations from insights")
print("\n" + "="*60)
