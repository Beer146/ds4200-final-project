import json

# This script organizes your code into clean notebook cells
cells_content = [
    # CELL 1: Imports and Helper Function
    r"""import kagglehub
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

# --- Helper Function ---
def save_matplotlib_to_html(fig, filename):
    '''
    Converts a Matplotlib figure to a base64 string and embeds it 
    into an HTML file.
    '''
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f'''
    <html>
    <head><title>{filename}</title></head>
    <body style="text-align:center;">
        <h1>{filename.replace('.html', '').replace('_', ' ').title()}</h1>
        <img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;" />
    </body>
    </html>
    '''
    
    with open(filename, 'w') as f:
        f.write(html_content)
    print(f"Successfully saved: {filename}")
    plt.close(fig)
""",

    # CELL 2: Data Loading
    r"""print("Downloading and processing data...")
path = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")

files_f = ['HDFCBANK.csv', 'ICICIBANK.csv', 'SBIN.csv', 'KOTAKBANK.csv']

dff = pd.DataFrame()

for x in files_f:
    df_temp = pd.read_csv(f"{path}/{x}")
    
    if 'Symbol' not in df_temp.columns:
        df_temp['Symbol'] = x.replace('.csv', '')
    
    dff = pd.concat([dff, df_temp], ignore_index=True)

# Processing
dff['Sector'] = 'Financial'
dff['Date'] = pd.to_datetime(dff['Date'])
dff = dff.sort_values(['Symbol', 'Date'])

# Filter Date Range
start_date = '2004-01-01'
end_date = '2021-04-30'
dff = dff[(dff['Date'] >= start_date) & (dff['Date'] <= end_date)]
""",

    # CELL 3: Calculations
    r"""# Metrics
dff['Daily_R'] = dff.groupby('Symbol')['Close'].pct_change()

# Rolling Volatility (60-day window)
dff['Volatility'] = dff.groupby('Symbol')['Daily_R'].transform(
    lambda x: x.rolling(window=60).std() * np.sqrt(252)
)

# Annual Returns 
dff['Year'] = dff['Date'].dt.year
annual_returns = dff.groupby(['Symbol', 'Year'])['Close'].agg(['first', 'last']).reset_index()
annual_returns['Return'] = ((annual_returns['last'] - annual_returns['first']) / annual_returns['first']) * 100

# Setup Plot Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

stock_colors = {'HDFCBANK': '#1f77b4', 'ICICIBANK': '#ff7f0e', 'SBIN': '#2ca02c', 'KOTAKBANK': '#d62728'}
stocks = dff['Symbol'].unique()
""",

    # CELL 4: Viz 1
    r"""print("Generating Rolling Volatility Chart...")
fig1 = plt.figure(figsize=(16, 8)) 
for stock in stocks:
    stock_data = dff[dff['Symbol'] == stock]
    plt.plot(stock_data['Date'], stock_data['Volatility'] * 100, 
             label=stock, alpha=0.8, linewidth=1.5, color=stock_colors.get(stock))

plt.title('Financial Sector: 60-Day Rolling Volatility (Risk)', fontsize=16, fontweight='bold')
plt.ylabel('Annualized Volatility (%)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

save_matplotlib_to_html(fig1, "financial_volatility.html")
""",

    # CELL 5: Viz 2
    r"""print("Generating Individual Price Trends...")
fig2, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, stock in enumerate(stocks):
    if i < 4: 
        ax = axes[i]
        stock_data = dff[dff['Symbol'] == stock]
        ax.plot(stock_data['Date'], stock_data['Close'], color=stock_colors.get(stock), linewidth=2)
        ax.set_title(f"{stock} Stock Price", fontsize=14, fontweight='bold')
        ax.set_ylabel("Price (INR)")
        ax.grid(True, alpha=0.3)

plt.suptitle('Financial Sector: Individual Stock Price History', fontsize=20, y=1.02)
plt.tight_layout()

save_matplotlib_to_html(fig2, "financial_stock_prices.html")
""",

    # CELL 6: Viz 3
    r"""print("Generating Trading Volume Trends...")
dff_monthly = dff.set_index('Date').groupby('Symbol').resample('ME')['Volume'].mean().reset_index()

fig3 = plt.figure(figsize=(16, 7))
for stock in stocks:
    stock_data = dff_monthly[dff_monthly['Symbol'] == stock]
    plt.plot(stock_data['Date'], stock_data['Volume'], 
             label=stock, color=stock_colors.get(stock), alpha=0.8)

plt.title('Financial Sector: Average Monthly Trading Volume', fontsize=16, fontweight='bold')
plt.ylabel('Volume (Shares)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

save_matplotlib_to_html(fig3, "financial_volume.html")
""",

    # CELL 7: Viz 4
    r"""print("Generating Interactive Annual Returns...")
selection = alt.selection_point(fields=['Symbol'], bind='legend')

annual_chart = alt.Chart(annual_returns).mark_bar().encode(
    x=alt.X('Year:O', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Return:Q', title='Annual Return (%)'),
    color=alt.condition(
        selection,
        alt.Color('Symbol:N', scale=alt.Scale(domain=list(stock_colors.keys()), range=list(stock_colors.values()))),
        alt.value('lightgray')
    ),
    tooltip=['Year', 'Symbol', alt.Tooltip('Return', format='.2f')]
).properties(
    title='Financial Sector: Annual Returns (Click Legend to Filter)',
    width=600,
    height=400
).add_params(selection)

annual_chart.save('financial_annual_returns.html')
print("Successfully saved: financial_annual_returns.html")
""",

    # CELL 8: Viz 5
    r"""print("Generating Risk vs. Return Analysis...")

risk_return_data = dff.groupby('Symbol').agg({
    'Daily_R': ['mean', 'std'],
    'Close': ['first', 'last']
}).reset_index()

risk_return_data.columns = ['Symbol', 'Mean_Daily_Return', 'Daily_Std', 'First_Price', 'Last_Price']

risk_return_data['Total Return (%)'] = ((risk_return_data['Last_Price'] - risk_return_data['First_Price']) / risk_return_data['First_Price']) * 100
risk_return_data['Avg Volatility (%)'] = risk_return_data['Daily_Std'] * np.sqrt(252) * 100

scatter_chart = alt.Chart(risk_return_data).mark_circle(size=400).encode(
    x=alt.X('Avg Volatility (%)', title='Average Risk (Volatility %)'),
    y=alt.Y('Total Return (%)', title='Total Reward (Return %)'),
    color=alt.Color('Symbol', scale=alt.Scale(domain=list(stock_colors.keys()), range=list(stock_colors.values()))),
    tooltip=['Symbol', 'Total Return (%)', 'Avg Volatility (%)']
).properties(
    title='Risk vs. Return Profile (Higher & Left is Better)',
    width=600,
    height=400
).interactive()

text = scatter_chart.mark_text(align='left', baseline='middle', dx=15).encode(text='Symbol')

final_chart = scatter_chart + text
final_chart.save('financial_risk_return.html')
print("Successfully saved: financial_risk_return.html")
print("\nAll visualizations have been saved to the current directory.")
"""
]

# Create the notebook JSON structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Fill the notebook with cells
for content in cells_content:
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.splitlines(keepends=True)
    })

# Write to file
filename = "financial_analysis.ipynb"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Success! Created {filename}")
print("You can now open this file in Jupyter Notebook or VS Code.")