# import multiprocessing
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import warnings
from scipy.stats import genextreme
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

FOLDS = 5

columns_to_drop = [
        'fiscalDateEnding1',
        'fiscalDateEnding2',
        'open',
        'prev_close',
        'marketCap',
        'sharesOutstanding',
        'trailingEPS',
        'PEratio',
        'totalNetIncomePastYr',
        'marketCapYesterday',
        'cashDeltaY',
        'PEratio2'
        ]

columns_to_clean = [
        'marketCapChangeY'
        ]

percent_cols = [
        '10YR',
    ]

target = 'marketCapChangeY'

def clean_dataframe(dataframe, columns_to_check):
    # Copy the original DataFrame to avoid modifying it directly
    cleaned_df = dataframe.copy()
    # Iterate through each specified column
    for column in columns_to_check:
        # Remove rows with inf values
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
        # Reveal nan indices
        # nan_indices = cleaned_df.index[cleaned_df.isnull().any(axis=1)].tolist()
        # Remove rows with NaN values in specified columns
        cleaned_df = cleaned_df.dropna(subset=[column])
        cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def importanceGraph(model,columns):
    importances = model.feature_importances_
    importance_data = pd.Series(importances, index=columns).sort_values()
    importance_data.plot(kind='bar')
    plt.xlabel('Feature')
    plt.ylabel('Importance %')
    plt.title('Importance Percentage of Different Financial Metrics')
    plt.tight_layout()
    plt.show()

def graphLosses(eval_result):
    train = eval_result['validation_0']['mae']
    test = eval_result['validation_1']['mae']
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='Training Loss')
    plt.plot(test, label='Testing Loss')
    plt.xlabel('Rounds')
    plt.ylabel('MAE Loss')
    plt.title('Loss Over Time')
    plt.legend()
    plt.show()

def runKFold(training_data_x, training_data_y, groups, model):

    scores = []
    gkf = GroupKFold(n_splits=FOLDS)

    for train, test in gkf.split(training_data_x,training_data_y,groups=groups):

        X_train_fold = training_data_x.iloc[train]
        X_val_fold = training_data_x.iloc[test]
        Y_train_fold = training_data_y.iloc[train]
        Y_val_fold = training_data_y.iloc[test]

        #scale features for x
        scalerx = RobustScaler()
        scalerx.fit(X_train_fold)
        np_train_x = scalerx.transform(X_train_fold)
        np_val_x = scalerx.transform(X_val_fold)

        scalery = RobustScaler()
        scalery.fit(Y_train_fold)
        np_train_y = scalery.transform(Y_train_fold)
        np_val_y = scalery.transform(Y_val_fold)

        eval_set = [(np_val_x, np_val_y)]
        model.fit(np_train_x, np_train_y, early_stopping_rounds=25, eval_set=eval_set, eval_metric="mae", verbose=False)

        scaled_pred = model.predict(np_val_x)
        #real_prediction = np.exp(scalery.inverse_transform(scaled_pred.reshape(-1, 1)))
        #real_target = np.exp(scalery.inverse_transform(np_val_y))
        real_prediction = scalery.inverse_transform(scaled_pred.reshape(-1, 1))
        real_target = scalery.inverse_transform(np_val_y)
        mae = np.mean(np.abs(real_prediction - real_target))
        #rmse = np.sqrt(np.mean((real_prediction - real_target) ** 2))
        #mape = np.mean(np.abs((real_target - real_prediction) / real_target)) * 100

        scores.append(mae)

    avg_score = np.mean(scores)
    return avg_score

def createModel(ticker=None):

    file_path = 'dataset.csv'
    dataset = pd.read_csv(file_path)

    #drop features when needed
    dataset = dataset.drop(columns_to_drop,
                         axis=1)

    #Get percent changes of columns
    dataset[percent_cols] = dataset.groupby('ticker')[percent_cols].pct_change()
    dataset = clean_dataframe(dataset, percent_cols)

    # dataset[target] = dataset[target].abs()
    # dataset[target] = np.log(dataset[target])

    #Clean data
    dataset = clean_dataframe(dataset, columns_to_clean)

    #remove a ticker for training/testing if needed
    if(ticker == None):
        print('No ticker provided. Continuing as normal.')
    else:
        result = dataset[dataset['ticker'] == ticker]
        if result.empty:
            print('Ticker not found in dataset. Continuing as normal.')
        else:
            print('Removing ticker from dataset and training/testing.')
            ticker_dataset = dataset[dataset['ticker'] == ticker]
            dataset = dataset[dataset['ticker'] != ticker]
            dataset.reset_index(drop=True, inplace=True)
            ticker_dataset.reset_index(drop=True, inplace=True)

    #split on tickers
    splitter = GroupShuffleSplit(test_size=(1/FOLDS), n_splits=2, random_state=50)
    split = splitter.split(dataset,groups=dataset['ticker'])
    train_inds,test_inds = next(split)


    #setup training data for KFold
    train = dataset.iloc[train_inds]
    train_x = train.drop(['ticker', target, 'reportedDate'],
                     axis=1)
    train_groups = train['ticker']
    train_groups = train_groups.to_frame()
    train_y = train[target]
    train_y = train_y.to_frame()


    #test data
    test = dataset.iloc[test_inds]
    test_x = test.drop(['ticker', target,'reportedDate'],
                         axis=1)
    test_y = test[target]
    test_y = test_y.to_frame()
    test_groups = test['ticker']
    test_groups = test_groups.to_frame()

    # hyper parameter tuning
    max_depth_grid = [None, 3, 5, 10]
    learning_rate_grid = [.01, .05, .1, .2]
    subsample_grid = [.5, .6, .7, .8, .9]
    colsampletree_grid = [0.6, 0.75, 1]
    best_depth = 0
    best_learningrate = 0
    best_subsample = 0
    best_col = 0
    top_score = 0
    config = 1

    for i in range(len(max_depth_grid)):
        for j in range(len(learning_rate_grid)):
            for k in range(len(subsample_grid)):
                for l in range(len(colsampletree_grid)):
                    print('Testing configuration: ' + str(config) + '/240')
                    model = XGBRegressor(max_depth=max_depth_grid[i], learning_rate=learning_rate_grid[j],
                                     n_estimators=500, subsample=subsample_grid[k], colsample_bytree = colsampletree_grid[l] , random_state=42)
                    score = runKFold(train_x, train_y, train_groups, model)
                # print(score)
                    if (i == 0 and j == 0 and k == 0):
                        top_score = score
                    elif (score < top_score):
                        top_score = score
                        best_depth = i
                        best_learningrate = j
                        best_subsample = k
                        best_col = l

                    config+=1

    # display results
    print("Minimum adjusted score on training data: " + str(top_score))
    print("With depth: " + str(max_depth_grid[best_depth]))
    print("With learning rate: " + str(learning_rate_grid[best_learningrate]))
    print("With subsample fraction: " + str(subsample_grid[best_subsample]))
    print("With colsample by tree: " + str(colsampletree_grid[best_col]) + "\n")

    # create and train model from optimal hyperparameters parameters
    validated_model = XGBRegressor(max_depth=max_depth_grid[best_depth],
                                   learning_rate=learning_rate_grid[best_learningrate],
                                   n_estimators=500, subsample=subsample_grid[best_subsample], colsample_bytree=colsampletree_grid[best_col], random_state=42)
    # scale features
    scalerx = RobustScaler()
    scalerx.fit(train_x)
    np_train_x = scalerx.transform(train_x)
    np_val_x = scalerx.transform(test_x)

    scalery = RobustScaler()
    scalery.fit(train_y)
    np_train_y = scalery.transform(train_y)
    np_val_y = scalery.transform(test_y)

    eval_set = [(np_train_x, np_train_y), (np_val_x, np_val_y)]
    validated_model.fit(np_train_x, np_train_y, eval_set=eval_set, eval_metric="mae", early_stopping_rounds=25, verbose=False)
    evals = validated_model.evals_result()

    # generate evaluation graphs
    graphLosses(evals)

    # generate importance of features graph
    importanceGraph(validated_model, train_x.columns)

    scaled_pred = validated_model.predict(np_val_x)

    #real_prediction = np.exp(scalery.inverse_transform(scaled_pred.reshape(-1, 1)))
    #real_target = np.exp(scalery.inverse_transform(np_val_y))
    real_prediction = scalery.inverse_transform(scaled_pred.reshape(-1, 1))
    real_target = scalery.inverse_transform(np_val_y)

    #overall_rmse = np.sqrt(np.mean((real_prediction - real_target) ** 2))
    overall_mae = np.mean(np.abs(real_prediction - real_target))
    #overall_mape = np.mean(np.abs((real_target - real_prediction) / real_target)) * 100

    print('Minimum adjusted score on validation data: ' + str(overall_mae))

    #calculate rmse distribution by stock
    test_targets_df = pd.DataFrame(real_target, columns=['Target'])
    test_predictions_df = pd.DataFrame(real_prediction, columns=['Prediction'])
    test_groups = test_groups.reset_index(drop=True)
    test_groups = pd.concat([test_groups, test_targets_df], axis=1)
    test_groups = pd.concat([test_groups, test_predictions_df], axis=1)

    unique_tickers = test_groups['ticker'].unique().tolist()
    scores_lst = []
    overpredict_lst = []

    for i in range(len(unique_tickers)):
        condition = test_groups['ticker'] == unique_tickers[i]
        targets = np.array(test_groups.loc[condition, 'Target'])
        count = 0
        predictions = np.array(test_groups.loc[condition, 'Prediction'])
        #stock_rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        stock_mae = np.mean(np.abs(targets-predictions))
        for i in range(len(targets)):
            if predictions[i] > targets[i]:
                count+=1
        overpredict_lst.append(count / len(targets))
        #stock_mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        scores_lst.append(stock_mae)

    # max_index = np.argmax(scores_lst)
    # std_dev = np.std(scores_lst)
    # mean = np.mean(scores_lst)
    # print('Average stock score: ' + str(mean))
    # print('Standard deviation: ' + str(std_dev))
    # print('Stock with highest loss: ' + unique_tickers[max_index])
    # print('With score: ' + str(scores_lst[max_index]) + '\n')
    #
    # #get_best_distribution(scores_lst)

    max_index = np.argmax(overpredict_lst)
    std_dev = np.std(overpredict_lst)
    mean = np.mean(overpredict_lst)
    print('Average stock score: ' + str(mean))
    print('Standard deviation: ' + str(std_dev))
    print('Stock with highest percentage of overpredictions: ' + unique_tickers[max_index])
    print('With score: ' + str(overpredict_lst[max_index]) + '\n\n')
    print('Total of: ' + str(len(unique_tickers)) + ' tickers.')

    return validated_model, scalerx, scalery

#Given a dataset, a model and scalers, generate predictions
def predictStocks(tested_model, Xscaler, Yscaler):
    file_path = 'datasetProduction.csv'
    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(columns_to_drop,
                           axis=1)

    dataset[percent_cols] = dataset.groupby('ticker')[percent_cols].pct_change()
    dataset = clean_dataframe(dataset, percent_cols)

    #if predicting p/e
    # dataset[target] = dataset[target].abs()
    # dataset[target] = np.log(dataset[target])

    #setup for training
    x = dataset.drop(['ticker', target,'reportedDate'],
                            axis=1)

    np_x = Xscaler.transform(x)

    scaled_stock_pred = tested_model.predict(np_x)

    #select scale method
    stock_prediction = Yscaler.inverse_transform(scaled_stock_pred.reshape(-1, 1))
    #stock_prediction = np.exp(Yscaler.inverse_transform(scaled_stock_pred.reshape(-1, 1)))

    # append onto the stock dataframe
    stock_predictions_df = pd.DataFrame(stock_prediction, columns=['Prediction'])
    ticker_dataset = pd.concat([dataset, stock_predictions_df], axis=1)

    graphDataset(ticker_dataset)


#Given a dataset with predictions, graph the predicted vs. the actual value
def graphDataset(dataset):

    unique_tickers = dataset['ticker'].unique().tolist()

    for ticker in unique_tickers:

        plt.figure(figsize=(15, 10))

        x = dataset.loc[dataset['ticker'] == ticker, target].tolist()
        y = dataset.loc[dataset['ticker'] == ticker, 'Prediction'].tolist()
        dates = dataset.loc[dataset['ticker'] == ticker, 'reportedDate'].tolist()

        predictions = np.array(y)
        targets = np.array(x)

        #rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        #mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        #mae = np.mean(np.abs(targets-predictions))
        # print(ticker + ' MAE:' + str(mae))

        count = 0
        for i in range(len(targets)):
            if predictions[i] > targets[i]:
                count += 1
        print(ticker + ' overpredict %:' + str(count / len(targets)))

        # #graph data
        # plt.plot(dates, predictions, marker='o', label='Predicted: ' + target)
        # plt.plot(dates, targets, marker='o', label='Acutal: ' + target)
        # plt.title(ticker)
        # plt.xlabel('Date')
        # plt.ylabel(target)
        # plt.xticks(rotation='vertical')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        #
        # #observation_count = len(x)
        # #legend_entry = f'Data points (n={observation_count})'
        # #plt.legend(handles=[scatter], labels=[legend_entry])
        #
        # plt.show()

        y_min = min(y)
        y_max = max(y)

        scatter = plt.scatter(x, y)
        plt.title(ticker)
        plt.xlabel('Actual: ' + target)
        plt.ylabel('Predicted: ' + target)
        plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line through y=0
        plt.axvline(0, color='black', linewidth=0.5)
        plt.plot(x, x, color='red', linestyle='--')

        plt.ylim(y_min - 2, y_max + 2)

        observation_count = len(x)
        legend_entry = f'Data points (n={observation_count})'
        plt.legend(handles=[scatter], labels=[legend_entry])

        plt.show()

model,scalerx,scalery = createModel()
predictStocks(model,scalerx,scalery)