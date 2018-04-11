"""Implement a market simulator that processes a dataframe instead of 
a csv file.
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from analysis import get_portfolio_value, get_portfolio_stats, \
plot_normalized_data
from util import get_data, normalize_data


def compute_portvals_single_symbol(df_orders, symbol, start_val=1000000, 
    commission=9.95, impact=0.005):
    """Compute portfolio values for a single symbol.

    Parameters:
    df_orders: A dataframe with orders for buying or selling stocks. There is
    no value for cash (i.e. 0).
    symbol: The stock symbol whose portfolio values need to be computed
    start_val: The starting value of the portfolio (initial cash available)
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the 
    historical data at each transaction
    
    Returns:
    portvals: A dataframe with one column containing the value of the portfolio
    for each trading day
    """

    # Sort the orders dataframe by date
    df_orders.sort_index(ascending=True, inplace=True)
    
    # Get the start and end dates
    start_date = df_orders.index.min()
    end_date = df_orders.index.max()

    # Create a dataframe with adjusted close prices for the symbol and for cash
    df_prices = get_data([symbol], pd.date_range(start_date, end_date))
    del df_prices["SPY"]
    df_prices["cash"] = 1.0

    # Fill NAN values if any
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices.fillna(1.0, inplace=True)

    # Create a dataframe that represents changes in the number of shares by day
    df_trades = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, 
        df_prices.columns)
    for index, row in df_orders.iterrows():
        # Total value of shares purchased or sold
        traded_share_value = df_prices.loc[index, symbol] * row["Shares"]
        # Transaction cost 
        transaction_cost = commission + impact * df_prices.loc[index, symbol] \
                            * abs(row["Shares"])

        # Update the number of shares and cash based on the type of transaction
        # Note: The same asset may be traded more than once on a particular day
        # If the shares were bought
        if row["Shares"] > 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
        # If the shares were sold
        elif row["Shares"] < 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
    # Create a dataframe that represents on each particular day how much of
    # each asset in the portfolio
    df_holdings = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, 
        df_prices.columns)
    for row_count in range(len(df_holdings)):
        # In the first row, the number shares are the same as in df_trades, 
        # but start_val must be added to cash
        if row_count == 0:
            df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()
            df_holdings.iloc[0, -1] = df_trades.iloc[0, -1] + start_val
        # The rest of the rows show cumulative values
        else:
            df_holdings.iloc[row_count] = df_holdings.iloc[row_count-1] \
                                            + df_trades.iloc[row_count]
        row_count += 1

    # Create a dataframe that represents the monetary value of each asset 
    df_value = df_prices * df_holdings
    
    # Create portvals dataframe
    portvals = pd.DataFrame(df_value.sum(axis=1), df_value.index, ["port_val"])
    return portvals

