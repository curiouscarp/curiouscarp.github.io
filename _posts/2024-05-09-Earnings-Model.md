---
Title: Earnings Model
Date: 2024-05-09
Layout: post
---

I was previously working on a model to predict vol adjusted price changes using data before and after an earnings release. 
If earnings are released at t+1 (morning or evening), I looked at the percentage change in price using the close at time t, and the open at time t+2.
I then realized a much longer time window could be appropriate at the cost of fewer datapoints.
For example, it could be interesting to look at adjusted returns for the SP500 stocks relative to meme stocks for the past few years.
I use a sliding window to calculate annualized standard deviation for the volatility adjustment, beta, and asset correlations. 
An unanchored window could be useful. 
I am currently testing different window sizes. 
Basically, I'm just trying stuff.

For other variables, I consider the changes in cash and cash equivalents, working capital, net income, total revenue, cash flow, EBITDA, debt, EBITDA to sales, and operating income to sales. 
For macro related variables I have the yearly change in GDP.
I am working on a new method to include the change in the 10 year. 
Multiplying the stock-bond correlation (SBC) and the 10 year could match the signs. 
For example, if the SBC is close to 0, then the impact of the 10 year on stock prices would be negligible.

Also, it looks like treasury issuance tends to already be priced into the market.

