"""Implement a StrategyLearner that trains a QLearner for trading a symbol."""

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from util import get_data, create_df_benchmark, create_df_trades
import QLearner as ql
from indicators import get_momentum, get_sma_indicator, compute_bollinger_value
from marketsim import compute_portvals_single_symbol, market_simulator
from analysis import get_portfolio_stats


class StrategyLearner(object):
    # Constants for positions and order signals
    LONG = 1
    CASH = 0
    SHORT = -1

    def __init__(self, num_shares=1000, epochs=100, num_steps=10, 
                 impact=0.0, commission=0.00, verbose=False, **kwargs):
        """Instantiate a StrategLearner that can learn a trading policy.

        Parameters:
        num_shares: The number of shares that can be traded in one order
        epochs: The number of times to train the QLearner
        num_steps: The number of steps used in getting thresholds for the
        discretization process. It is the number of groups to put data into.
        impact: The amount the price moves against the trader compared to the
        historical data at each transaction
        commission: The fixed amount in dollars charged for each transaction
        verbose: If True, print and plot data in add_evidence
        **kwargs: Arguments for QLearner
        """
        self.epochs = epochs
        self.num_steps = num_steps
        self.num_shares = num_shares
        self.impact = impact
        self.commission = commission
        self.verbose = verbose
        # Initialize a QLearner
        self.q_learner = ql.QLearner(**kwargs)

    def get_features(self, prices):
        """Compute technical indicators and use them as features to be fed
        into a Q-learner.
        
        Parameters:
        prices: Adjusted close prices of the given symbol
        
        Returns:
        df_features: A pandas dataframe of the technical indicators
        """
        window = 10
        # Compute rolling mean
        rolling_mean = prices.rolling(window=window).mean()
        # Compute_rolling_std
        rolling_std = prices.rolling(window=window).std()
        # Compute momentum
        momentum = get_momentum(prices, window)
        # Compute SMA indicator
        sma_indicator = get_sma_indicator(prices, rolling_mean)
        # Compute Bollinger value
        bollinger_val = compute_bollinger_value(prices, rolling_mean, rolling_std)
        # Create a dataframe with three features
        df_features = pd.concat([momentum, sma_indicator], axis=1)
        df_features = pd.concat([df_features, bollinger_val], axis=1)
        df_features.columns = ["ind{}".format(i) 
                                for i in range(len(df_features.columns))]
        df_features.dropna(inplace=True)
        return df_features
