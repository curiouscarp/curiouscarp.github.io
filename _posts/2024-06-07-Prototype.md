---
Title: Prototype
Date: 2024-06-11
Layout: post
---

So, I’ve widened out the signal which makes way more sense in hindsight. 
I use the adjusted close prices to account for dividends and share splits. 
The dataset I built uses three-year percent changes from the latest quarterly earnings report. 
For example, Visa had an earnings release January 25th, 2024. 
The annualized vol adjusted close from 3 years ago is about 113%.
For the features I’ll look at how much quarterly total revenue grew over the 3 years, for example. 

Since there is a single time window with the new approach, macro variables would account for little variation across samples. 
I trained the data on a sample of 410 stocks in the SP500. 
For the machine learning I use XGBoost with 5 folds for cross validation. 
I use a grid search for the learning rate, subsample, and maxdepth. 
To make the model better, one might be able to predict sharpe ratios as a gauge for risk adjusted return. 
That said, the dividends add complexity, since I wouldn't just be looking at stock returns. 
There would have to be a total return and the adjustments to the price series data are non trivial.
I am looking at additional Python libraries to tackle the problem.
Wall Street is almost comically oblivious.
As per friends or other readers, I would like to see everyone as rich as possible.

Figure 1: Sample Features

![features](/assets/images/featuresNew.png)

Figure 2: Losses

![losses](/assets/images/lossesNew.png)

Figure 3a: AMC

![amc.png](/assets/images/amcNew.png)
