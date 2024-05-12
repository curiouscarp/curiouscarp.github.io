---
Title: Earnings Model
Date: 2024-05-09
Layout: post
---

I am working on a model to predict vol adjusted price changes using data before and after an earnings release. 
If earnings are released at t+1 (morning or evening), I look at the percentage change in price using the close at time t, and the open at time t+2. 
I’ve also added a time variable.
Quarterly changes in price could be useful to consider as well.
Enterprise value (EV) is also considered a more holistic view than market cap (MC).
Such an approach would likely require some smoothing given the high variability in debt and cash. 

Right now, I use a sliding window to calculate annualized standard deviation for the volatility adjustment. 
I also use the same sliding window to generate covariance and variance for beta. 
An unanchored window could be useful. 
I am currently testing different window sizes. 

For other variables, I consider the yearly change in cash and cash equivalents, working capital, net income, total revenue, cash flow, EBITDA, debt, EBITDA to sales, and operating income to sales. 
I also include the most recent raw values for the 10 year, and the VIX. For macro related variables I have the yearly change in GDP and CPI (CPILFESL from FRED). 
I also have an earnings surprise percentage.

I am incorporating technical indicators as well. 
I am not a chart technician, so I am trying to figure out the best indicators and how to use them appropriately. 
The current strategy, albeit rudimentary, is to look at the value of these the day before the earnings release. 
Right now, I look at mama-fama for a trend direction, adx for trend strength, rsi, stochrsi, mfi, and apo. 
I binary encode the variables. It could be useful to incorporate some kind of indicator of a recent crossover, though I am not sure how long daily technical buy/sell signals are good for.
It's probably important to remember that more predictors do not always lead to a better model.
Predictors can also correlate with each other. 
If any readers have suggestions/ideas I am open.

Also, it looks like treasury issuance tends to already be priced into the market.

