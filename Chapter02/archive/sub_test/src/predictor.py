# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm
import os
import pickle
import sys
import warnings
from glob import glob
import re
import datetime
import itertools
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.stochastic import percent_k as srv_k
from pyti.stochastic import percent_d as srv_d
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DayLocator, DateFormatter
#from mpl_finance import candlestick2_ohlc, volume_overlay
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import spearmanr


class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"].copy()
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]
            # 日付列をpd.Timestamp型に変換してindexに設定
            stock_labels["datetime"] = pd.to_datetime(stock_labels["base_date"])
            stock_labels.set_index("datetime", inplace=True)

            # 特定の目的変数に絞る
            labels = stock_labels[label]
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END].copy()
                _val_X = feats[cls.VAL_START : cls.VAL_END].copy()
                _test_X = feats[cls.TEST_START :].copy()

                _train_y = labels[: cls.TRAIN_END].copy()
                _val_y = labels[cls.VAL_START : cls.VAL_END].copy()
                _test_y = labels[cls.TEST_START :].copy()

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    @classmethod
    def cross_X(cls,x):
        return np.prod(x)
        
    @classmethod
    def get_features_for_predict(cls,dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # おおまかな手順の1つ目
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"]
        periods = [10, 20, 40]
        # 特定の銘柄コードのデータに絞る
        stock_fin = stock_fin[stock_fin["Local Code"] == code]
        fin_data = stock_fin[~stock_fin.duplicated(subset=['Local Code', 'Result_FinancialStatement ReportType',"Result_FinancialStatement FiscalYear"],keep='last')]
        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        
        fin_data = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        seasons = stock_fin["Result_FinancialStatement ReportType"].unique()
        columns = fin_data.columns
        columns = columns.to_list()
        #columns_list = ["Result_FinancialStatement NetSales","Result_FinancialStatement OrdinaryIncome","Result_FinancialStatement TotalAssets","Result_FinancialStatement NetAssets"]
        columns_list = ["Result_FinancialStatement NetSales","Result_FinancialStatement OrdinaryIncome","Result_FinancialStatement TotalAssets","Result_FinancialStatement NetAssets",
                   'Result_Dividend QuarterlyDividendPerShare','Result_Dividend AnnualDividendPerShare']
        for column in columns_list:
            a = "last "+column
    #         print(a)
    #         print(type(columns))
            columns.append(a)
        df_result = pd.DataFrame(index=[], columns=columns)
        # columns_list.append("base_date")
        # columns_list.append("Local Code")
        for season in seasons:
            #df["last "+column] = 0
            #print(columns_list)
            df_test = fin_data[fin_data["Result_FinancialStatement ReportType"]==season].copy()
            for column in columns_list:
                #print(columns)
                df_test["last "+column] = df_test[column]
                df_test["last "+column] = df_test[column].shift()
                #df = pd.merge(df,df_test[["last "+column,"base_date","Local Code"]],on = ["base_date","Local Code"],how="left")
                #df_ab, df_ac, on='a', how='left'
            #print(df_result)
            #print(df_test)
            df_result = pd.concat([df_result,df_test])
        df_result["NetSales_growth_rate"] = df_result["Result_FinancialStatement NetSales"] / df_result["last Result_FinancialStatement NetSales"]
        df_result["OrdinaryIncome_growth_rate"] = df_result["Result_FinancialStatement OrdinaryIncome"] / df_result["last Result_FinancialStatement OrdinaryIncome"]
        df_result["TotalAssets_growth_rate"] = df_result["Result_FinancialStatement TotalAssets"] / df_result["last Result_FinancialStatement TotalAssets"]
        df_result["NetAssets_growth_rate"] = df_result["Result_FinancialStatement NetAssets"] / df_result["last Result_FinancialStatement NetAssets"]
        df_result["QuarterlyDividendPerShare_growth_rate"] = df_result["Result_Dividend QuarterlyDividendPerShare"] / df_result["last Result_Dividend QuarterlyDividendPerShare"]
        df_result["AnnualDividendPerShare_growth_rate"] = df_result["Result_Dividend AnnualDividendPerShare"] / df_result["last Result_Dividend AnnualDividendPerShare"]
        #df_result = df_result.drop(["EndOfDayQuote ExchangeOfficialClose","macd_hist_shift","stocas_hist_shift","stocas_huge_signal"], axis=1)
    #     # fin_dataのnp.float64のデータのみを取得
    #     fin_data = fin_data.select_dtypes(include=["float64"])
    #     # 欠損値処理
    #     fin_feats = fin_data.fillna(0)

        # おおまかな手順の2つ目
        # stock_priceデータを読み込む
        price = dfs["stock_price"]
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code]
        # 終値のみに絞る
        feats = price_data[["EndOfDayQuote ExchangeOfficialClose","EndOfDayQuote Volume"]]
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

        # 終値の20営業日リターン
        feats["return_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(20)
        # 終値の40営業日リターン
        feats["return_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(60)
        # 終値の20営業日ボラティリティ
        feats["volatility_0.5month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(10).std()
        )
        # 終値の40営業日ボラティリティ
        feats["volatility_1month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(20).std()
        )
        # 終値の60営業日ボラティリティ
        feats["volatility_2month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(40).std()
        )
        
        for period in periods:
            col = "5 windows volatility  {} mean".format(period)
            feats[col] = feats["volatility_0.5month"].rolling(period).mean()
        
        # ヒストリカル・ボラティリティ移動平均
        for period in periods:
            col = "25 windows volatility  {} mean".format(period)
            feats[col] = feats["volatility_1month"].rolling(period).mean()
            
        # ヒストリカル・ボラティリティ移動平均
        for period in periods:
            col = "75 windows volatility  {} mean".format(period)
            feats[col] = feats["volatility_2month"].rolling(period).mean()

        # ヒストリカル・ボラティリティ移動平均微分値
        for period in periods:
            col = "5 windows volatility  {} mean diff".format(period)
            feats[col] = feats["volatility_0.5month"].rolling(10).mean().pct_change(period)

        # ヒストリカル・ボラティリティ移動平均微分値
        for period in periods:
            col = "25 windows volatility  {} mean diff".format(period)
            feats[col] = feats["volatility_1month"].rolling(20).mean().pct_change(period)

        # ヒストリカル・ボラティリティ移動平均微分値
        for period in periods:
            col = "75 windows volatility  {} mean diff".format(period)
            feats[col] = feats["volatility_2month"].rolling(30).mean().pct_change(period)
        
        macd_period = {'long' : 26, 'short' : 12}
        sma_period  = 9
        feats['macd'] = macd(feats['EndOfDayQuote ExchangeOfficialClose'].values.tolist(), 12, 26)
        feats['macd_signal'] = sma(feats['macd'].values.tolist(), sma_period)
        feats["macd_hist"] = feats["macd"] - feats["macd_signal"]
        feats["macd_hist_shift"] = feats["macd_hist"].shift()
        feats.loc[feats["macd_hist"] < 0,"macd_hist_signal"] = -1
        feats.loc[feats["macd_hist"] > 0,"macd_hist_signal"] = 1
        feats.loc[feats["macd_hist"] == 0,"macd_hist_signal"] = 0
        feats["macd_cross_signal"] = feats["macd_hist"]*feats["macd_hist_shift"]
        feats.loc[feats["macd_cross_signal"] <= 0, "macd_cross_signal"] = 0
        feats.loc[feats["macd_cross_signal"] > 0, "macd_cross_signal"] = 1
        feats["macd_cross_sign_20"] = (1-feats["macd_cross_signal"].rolling(20).apply(cls.cross_X))*feats["macd_hist_signal"]
        feats["macd_cross_sign_10"] = (1-feats["macd_cross_signal"].rolling(10).apply(cls.cross_X))*feats["macd_hist_signal"]
        feats["macd_cross_sign_5"] = (1-feats["macd_cross_signal"].rolling(5).apply(cls.cross_X))*feats["macd_hist_signal"]
        #feats.loc[feats["macd_cross_sign"] > 0, "macd_cross_sign"] = 1
        mac_cols = ["macd","macd_signal","macd_hist"]
        mac_cross_cols = ["macd_cross_sign_20","macd_cross_sign_10"]
        feats["slow%k"] = srv_d(feats["EndOfDayQuote ExchangeOfficialClose"].values.tolist(), 14)*100
        feats["slow%d"] = feats["slow%k"].rolling(3).mean()
        feats["stocas_hist"] = feats["slow%k"] - feats["slow%d"]
        feats["stocas_hist_shift"] = feats["stocas_hist"].shift()
        feats["stocas_cross_signal"] = feats["stocas_hist"]*feats["stocas_hist_shift"]
        feats.loc[feats["stocas_cross_signal"] <= 0, "stocas_cross_signal"] = 0
        feats.loc[feats["stocas_cross_signal"] > 0, "stocas_cross_signal"] = 1
        feats.loc[feats["stocas_hist"] < 0,"stocas_hist_signal"] = -1
        feats.loc[feats["stocas_hist"] > 0,"stocas_hist_signal"] = 1
        feats.loc[feats["stocas_hist"] == 0,"stocas_hist_signal"] = 0
        feats["stocas_huge_signal"] = 0
        feats.loc[feats["slow%k"] <= 20,"stocas_huge_signal"] = 1
        feats.loc[feats["slow%k"] >= 80,"stocas_huge_signal"] = 1
        
        # feats["stocas_cross_sign_20"] = (1-feats["stocas_cross_signal"].rolling(20).apply(cross_X))*feats["stocas_hist_signal"]*feats["stocas_huge_signal"]
        # feats["stocas_cross_sign_10"] = (1-feats["stocas_cross_signal"].rolling(10).apply(cross_X))*feats["stocas_hist_signal"]*feats["stocas_huge_signal"]
        feats["stocas_cross_sign_5"] = (1-feats["stocas_cross_signal"].rolling(5).apply(cls.cross_X))*feats["stocas_hist_signal"]*feats["stocas_huge_signal"]
        stock_list = dfs["stock_list"]
        stock_data = stock_list[stock_list["Local Code"] == code]
        stock_data = stock_data[["33 Sector(Code)","17 Sector(Code)","IssuedShareEquityQuote IssuedShare","Size (New Index Series)"]]
        feats["IssuedShareEquityQuote IssuedShare"] = stock_data["IssuedShareEquityQuote IssuedShare"]
        #出来高移動平均線
        feats["Volume_5"] = feats["EndOfDayQuote Volume"].rolling(5).mean() / feats["IssuedShareEquityQuote IssuedShare"]
        feats["Volume_20"] = feats["EndOfDayQuote Volume"].rolling(20).mean() / feats["IssuedShareEquityQuote IssuedShare"]
        feats["Volume_40"] = feats["EndOfDayQuote Volume"].rolling(40).mean() / feats["IssuedShareEquityQuote IssuedShare"]
        
        #出来高移動平均線クロス
        feats["volumn_hist_5_20"] = feats["Volume_5"] - feats["Volume_20"]
        feats["volumn_hist_5_40"] = feats["Volume_5"] - feats["Volume_40"]
        feats["volumn_hist_20_40"] = feats["Volume_20"] - feats["Volume_40"]
        feats["volume_hist_shift"] = feats["volumn_hist_5_20"].shift()
        feats.loc[feats["volumn_hist_5_20"] < 0,"volume_hist_signal"] = -1
        feats.loc[feats["volumn_hist_5_20"] > 0,"volume_hist_signal"] = 1
        feats.loc[feats["volumn_hist_5_20"] == 0,"volume_hist_signal"] = 0
        feats["volume_cross_signal"] = feats["volumn_hist_5_20"]*feats["volume_hist_shift"]
        feats.loc[feats["volume_cross_signal"] <= 0, "volume_cross_signal"] = 0
        feats.loc[feats["volume_cross_signal"] > 0, "volume_cross_signal"] = 1
        feats["volume_cross_sign_20"] = (1-feats["volume_cross_signal"].rolling(20).apply(cls.cross_X,raw=True))*feats["volume_hist_signal"]
        feats["volume_cross_sign_10"] = (1-feats["volume_cross_signal"].rolling(10).apply(cls.cross_X,raw=True))*feats["volume_hist_signal"]
        feats["volume_cross_sign_5"] = (1-feats["volume_cross_signal"].rolling(5).apply(cls.cross_X,raw=True))*feats["volume_hist_signal"]
        
        # おおまかな手順の3つ目
        # 欠損値処理
        #feats = feats.fillna(0)
        # 元データのカラムを削除
        
        
        #財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(df_result.index)]
        df_result = df_result.loc[df_result.index.isin(feats.index)]
        feats = pd.merge(df_result,feats,left_index= True,right_index = True ,how = "left")
        
        sector_17_dict = {1:0.041404, 2:0.056027, 3:0.052955, 4:0.064411, 5:0.091106, 6:0.046410, 7:0.047056, 8:0.070079, 9:0.052416, 10:0.061940, 11:0.062365, 12:0.052021,
                        13:0.022329, 14:0.030389, 15:0.190156, 16:0.109204, 17:0.083384}
        sector_33_dict = {50:	0.026789,
                            1050: 0.075618,
                            2050: 0.052140,
                            3050: 0.042869,
                            3100: 0.044244,
                            3150: 0.035982,
                            3200: 0.074299,
                            3250: 0.091106,
                            3300: 0.043225,
                            3350: 0.076820,
                            3400: 0.060586,
                            3450: 0.047813,
                            3500: 0.045698,
                            3550: 0.050873,
                            3600: 0.070079,
                            3650: 0.050197,
                            3700: 0.042787,
                            3750: 0.060784,
                            3800: 0.043177,
                            4050: 0.062365,
                            5050: 0.059168,
                            5100: 0.024505,
                            5150: 0.063092,
                            5200: 0.047985,
                            5250: 0.069475,
                            6050: 0.022329,
                            6100: 0.030389,
                            7050: 0.190156,
                            7100: 0.131484,
                            7150: 0.061564,
                            7200: 0.125792,
                            8050: 0.083384,
                            9050: 0.061318}
    #     print("hello")
    #     print(stock_data["33 Sector(Code)"])
        topix_dict = {'TOPIX Small 2':1, 'TOPIX Mid400':3, 'TOPIX Small 1':2, '-':0,'TOPIX Large70':4, 'TOPIX Core30':5}
        feats["en_33"] = sector_33_dict[stock_data["33 Sector(Code)"].values[0]]
        feats["en_17"] = sector_17_dict[stock_data["17 Sector(Code)"].values[0]]
        feats["Ordinary_rate_of_return"] = feats["Result_FinancialStatement OrdinaryIncome"] / feats["Result_FinancialStatement NetSales"]

        feats["sector17's_Ordinary_rate_of_return_diff"] = feats["Ordinary_rate_of_return"] - feats["en_33"]
        feats["sector33's_Ordinary_rate_of_return_diff"] = feats["Ordinary_rate_of_return"] - feats["en_17"]
        feats["en_topix"] = topix_dict[stock_data["Size (New Index Series)"].values[0]]
        feats["IssuedShareEquityQuote IssuedShare"] = stock_data["IssuedShareEquityQuote IssuedShare"].values[0]
        feats["Net_income_per_stock"] = feats["Result_FinancialStatement NetIncome"] / feats["IssuedShareEquityQuote IssuedShare"]
        #stock_data = stock_data["en_33","en_17","IssuedShareEquityQuote IssuedShare"]
        feats["PBR"] =feats["EndOfDayQuote ExchangeOfficialClose"] * feats["IssuedShareEquityQuote IssuedShare"] / feats["Result_FinancialStatement NetAssets"]
        feats["stability"] = feats["Result_FinancialStatement NetAssets"] / feats["Result_FinancialStatement TotalAssets"]
        feats["ROE"] = feats["Result_FinancialStatement NetIncome"] / feats["Result_FinancialStatement NetAssets"]
        feats["ROA"] = feats["Result_FinancialStatement NetIncome"] / feats["Result_FinancialStatement TotalAssets"]
        feats.loc[feats['Result_FinancialStatement CashFlowsFromOperatingActivities'] <= 0, 'Operating_cash_flow'] = 0
        feats.loc[feats['Result_FinancialStatement CashFlowsFromOperatingActivities'] > 0, 'Operating_cash_flow'] = 4
        feats.loc[feats['Result_FinancialStatement CashFlowsFromFinancingActivities'] <= 0, 'Financial_cash_flow'] = 0
        feats.loc[feats['Result_FinancialStatement CashFlowsFromFinancingActivities'] > 0, 'Financial_cash_flow'] = 1
        feats.loc[feats['Result_FinancialStatement CashFlowsFromInvestingActivities'] <= 0, 'Investing_cash_flow'] = 0
        feats.loc[feats['Result_FinancialStatement CashFlowsFromInvestingActivities'] > 0, 'Investing_cash_flow'] = 2
        feats["cash_evaluation"] = feats["Operating_cash_flow"] + feats["Financial_cash_flow"] + feats["Investing_cash_flow"]
        
        #feats["forecast_OperatingIncome_growth_rate"] = feats["Forecast_FinancialStatement OperatingIncome"] / feats["Result_FinancialStatement OperatingIncome"]
        feats["forecast_NetSales_growth_rate"] = feats["Forecast_FinancialStatement NetSales"] / feats["Result_FinancialStatement NetSales"]
        feats["forecast_OrdinaryIncome_growth_rate"] = feats["Forecast_FinancialStatement OrdinaryIncome"] / feats["Result_FinancialStatement OrdinaryIncome"]
        feats["forecast_NetIncome_growth_rate"] = feats["Forecast_FinancialStatement NetIncome"] / feats["Result_FinancialStatement NetIncome"]
        feats['QuarterlyDividendPerShare'] = feats['Result_Dividend QuarterlyDividendPerShare'] / feats['EndOfDayQuote ExchangeOfficialClose'] 
        feats['AnnualDividendPerShare'] = feats['Result_Dividend AnnualDividendPerShare'] / feats['EndOfDayQuote ExchangeOfficialClose'] 
        
        feats['Forecast_QuarterlyDividendPerShare_growth_rate'] = feats['Forecast_Dividend QuarterlyDividendPerShare'] / feats['Result_Dividend QuarterlyDividendPerShare'] 
        feats['forecast_AnnualDividendPerShare_growth_rate'] = feats['Forecast_Dividend AnnualDividendPerShare'] / feats['Result_Dividend AnnualDividendPerShare']
        
        
        
        feats = feats.drop(["EndOfDayQuote ExchangeOfficialClose","macd_hist_shift","stocas_hist_shift","stocas_huge_signal","Operating_cash_flow","Operating_cash_flow","Financial_cash_flow",
                        "Investing_cash_flow","macd_cross_signal","Result_FinancialStatement FiscalYear","Forecast_FinancialStatement FiscalYear","Forecast_Dividend FiscalYear","Result_Dividend FiscalYear"
                        ,"en_33","en_17","Ordinary_rate_of_return","Result_FinancialStatement CashFlowsFromOperatingActivities","Result_FinancialStatement CashFlowsFromFinancingActivities",
                        "Result_FinancialStatement CashFlowsFromInvestingActivities","Result_FinancialStatement NetSales","Result_FinancialStatement OperatingIncome",
                        "Result_FinancialStatement OrdinaryIncome","Result_FinancialStatement NetIncome","Result_FinancialStatement TotalAssets","Result_FinancialStatement NetAssets",
                        "Result_FinancialStatement CashFlowsFromOperatingActivities","Result_FinancialStatement CashFlowsFromFinancingActivities","Forecast_FinancialStatement NetSales",
                        "Forecast_FinancialStatement OperatingIncome","Forecast_FinancialStatement OrdinaryIncome","Forecast_FinancialStatement NetIncome","Result_Dividend QuarterlyDividendPerShare",
                        "Result_Dividend AnnualDividendPerShare","last Result_FinancialStatement NetSales","last Result_FinancialStatement OrdinaryIncome","last Result_FinancialStatement TotalAssets"
                        ,"last Result_FinancialStatement NetAssets","last Result_Dividend QuarterlyDividendPerShare","last Result_Dividend AnnualDividendPerShare",
                        "Forecast_Dividend QuarterlyDividendPerShare","Forecast_Dividend AnnualDividendPerShare","volume_hist_shift","volume_hist_signal","volume_cross_signal","volume_hist_signal"], axis=1)
        

            


        feats = feats.select_dtypes(include=[int, float])
        feats = feats.astype('float64')
        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], 0)
        feats = feats.replace({'NetSales_growth_rate': np.nan, 'OrdinaryIncome_growth_rate': np.nan,"TotalAssets_growth_rate":np.nan,
                          "NetAssets_growth_rate":np.nan,"QuarterlyDividendPerShare_growth_rate":np.nan,"AnnualDividendPerShare_growth_rate":np.nan,
                          "forecast_NetSales_growth_rate":np.nan,"forecast_OrdinaryIncome_growth_rate":np.nan,"forecast_NetIncome_growth_rate":np.nan,
                          "Forecast_QuarterlyDividendPerShare_growth_rate":np.nan,"forecast_AnnualDividendPerShare_growth_rate":np.nan}, 1)
        # 銘柄コードを設定
        feats["code"] = code


        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X):
        # 特徴量グループを定義

        # テクニカル
        technical_cols = [
            x for x in train_X.columns if (x != "code")
        ]
        columns = {
            "technical": technical_cols,
        }
        return columns["technical"]

    @classmethod
    def create_model(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, _, _, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )
        # モデル作成
        model = RandomForestRegressor(random_state=0)
        model.fit(train_X, train_y)

        return model

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        try:
            for label in labels:
                m = os.path.join(model_path, f"my_model_{label}.pkl")
                with open(m, "rb") as f:
                    # pickle形式で保存されているモデルを読み込み
                    cls.models[label] = pickle.load(f)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(label)
            model = cls.create_model(cls.dfs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64
        column1 = feats.columns
        for column in column1:
            feats[column] = feats[column].fillna(feats[column].median())
        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]
        feature_columns = cls.get_feature_columns(cls.dfs, feats)
        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            #df[label] = cls.models[label].predict(feats)
            df[label] = cls.models[label].predict(feats[feature_columns])
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()
