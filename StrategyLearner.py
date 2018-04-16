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

    def get_thresholds(self, df_features, num_steps):
        """Compute the thresholds to be used in the discretization of features.
        thresholds is a 2-d numpy array where the first dimesion indicates the 
        indices of features in df_features and the second dimension refers to 
        the value of a feature at a particular threshold.
        """
        step_size = round(df_features.shape[0] / num_steps)
        df_copy = df_features.copy()
        thres = np.zeros(shape=(df_features.shape[1], num_steps))
        for i, feat in enumerate(df_features.columns):
            df_copy.sort_values(by=[feat], inplace=True)
            for step in range(num_steps):
                if step < num_steps - 1:
                    thres[i, step] = df_copy[feat].iloc[(step + 1) * step_size]
                # The last threshold must be = the largest value in df_copy
                else:
                    thres[i, step] = df_copy[feat].iloc[-1]
        return thres

    def discretize(self, df_features, non_neg_position, thresholds):
        """Discretize features and return a state.

        Parameters:
        df_features: The technical indicators to be discretized. They were  
        computed in get_features()
        non_neg_position: The position at the beginning of a particular day,
        before taking any action on that day. It is >= 0 so that state >= 0

        Returns:
        state: A state in the Q-table from which we will query for an action.
        It indicates an index of the first dimension in the Q-table
        """
        state = non_neg_position * pow(self.num_steps, len(df_features))
        for i in range(len(df_features)):
            thres = thresholds[i][thresholds[i] >= df_features[i]][0]
            thres_i = np.where(thresholds == thres)[1][0]
            state += thres_i * pow(self.num_steps, i)
        return state

    def get_position(self, old_pos, signal):
        """Find a new position based on the old position and the given signal.
        signal = action - 1; action is a result of querying a state, which was
        computed in discretize(), in the Q-table. An action is 0, 1 or 2. It is
        an index of the second dimension in the Q-table. We have to subtract 1
        from action to get a signal of -1, 0 or 1 (short, cash or long).
        """
        new_pos = self.CASH
        # If old_pos is not long and signal is to buy, new_pos will be long
        if old_pos < self.LONG and signal == self.LONG:
            new_pos = self.LONG
        # If old_pos is not short and signal is to sell, new_pos will be short
        elif old_pos > self.SHORT and signal == self.SHORT:
            new_pos = self.SHORT
        return new_pos

    def get_daily_reward(self, prev_price, curr_price, position):
        """Calculate the daily reward as a percentage change in prices: 
        - Position is long: if the price goes up (curr_price > prev_price),
          we get a positive reward; otherwise, we get a negative reward
        - Position is short: if the price goes down, we get a positive reward;
        otherwise, we a negative reward
        - Position is cash: we get no reward
        """
        return position * ((curr_price / prev_price) - 1)

    def has_converged(self, cum_returns, patience=10):
        """Check if the cumulative returns have converged.

        Paramters:
        cum_returns: A list of cumulative returns for respective epochs
        patience: The number of epochs with no improvement in cum_returns

        Returns: True if converged, False otherwise
        """
        # The number of epochs should be at least patience before checking
        # for convergence
        if patience > len(cum_returns):
            return False
        latest_returns = cum_returns[-patience:]
        # If all the latest returns are the same, return True
        if len(set(latest_returns)) == 1:
            return True
        max_return = max(cum_returns)
        # If one of recent returns improves, not yet converged
        if max_return in latest_returns:
            return False
        # If none of recent returns is greater than max_return, it has converged
        return (all(ret <= max_return for ret in latest_returns))

