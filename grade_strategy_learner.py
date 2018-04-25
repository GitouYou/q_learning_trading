"""Testing code for StrategyLearner.
Mostly based on the code provided by Georgia Tech, with the following edits:
- code edits to make the code compatible with Python 3x
- a lot of reformatting to make code readable and understandable
"""

import pytest
from grading import grader, GradeResult, run_with_timeout, IncorrectOutput

import os
import sys
import traceback as tb

import datetime as dt
import numpy as np
import pandas as pd
from collections import namedtuple

import time
import util
import random

# Module name to import
main_code = ["StrategyLearner",]

# Test cases
StrategyTestCase = namedtuple("Strategy", ["description","insample_args",
    "outsample_args","benchmark_type","benchmark","impact", "num_states", 
    "num_actions", "train_time","test_time","max_time","seed"])
strategy_test_cases = [
    StrategyTestCase(
        description="ML4T-220",
        insample_args=dict(symbol="ML4T-220", start_date=dt.datetime(2008,1,1),
                           end_date=dt.datetime(2009,12,31), start_val=100000),
        outsample_args=dict(symbol="ML4T-220", start_date=dt.datetime(2010,1,1),
                            end_date=dt.datetime(2011,12,31), start_val=100000),
        benchmark_type="clean",
        benchmark=1.0,
        impact=0.0,
        num_states=3000,
        num_actions=3,
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="AAPL",
        insample_args=dict(symbol="AAPL", start_date=dt.datetime(2008,1,1),
                           end_date=dt.datetime(2009,12,31),start_val=100000),
        outsample_args=dict(symbol="AAPL", start_date=dt.datetime(2010,1,1),
                            end_date=dt.datetime(2011,12,31),start_val=100000),
        benchmark_type="stock",
        benchmark=0.1581999999999999,
        impact=0.0,
        num_states=3000,
        num_actions=3,
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="SINE_FAST_NOISE",
        insample_args=dict(symbol="SINE_FAST_NOISE",
                           start_date=dt.datetime(2008,1,1),
                           end_date=dt.datetime(2009,12,31), start_val=100000),
        outsample_args=dict(symbol="SINE_FAST_NOISE",
                            start_date=dt.datetime(2010,1,1),
                            end_date=dt.datetime(2011,12,31), start_val=100000),
        benchmark_type="noisy",
        benchmark=2.0,
        impact=0.0,
        num_states=3000,
        num_actions=3,
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="UNH - In sample",
        insample_args=dict(symbol="UNH", start_date=dt.datetime(2008,1,1),
                           end_date=dt.datetime(2009,12,31),start_val=100000),
        outsample_args=dict(symbol="UNH", start_date=dt.datetime(2010,1,1),
                            end_date=dt.datetime(2011,12,31),start_val=100000),
        benchmark_type="stock",
        benchmark= -0.25239999999999996,
        impact=0.0,
        num_states=3000,
        num_actions=3,
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
]

max_points = 60.0 
# Surround comments with HTML <pre> tag (for T-Square comments field)
html_pre_block = True

MAX_HOLDINGS = 1000

# Test functon(s)
@pytest.mark.parametrize("description, insample_args, outsample_args, \
    benchmark_type, benchmark, impact, num_states, num_actions, train_time, \
    test_time, max_time, seed", strategy_test_cases)

def test_strategy(description, insample_args, outsample_args, benchmark_type, \
    benchmark, impact, num_states, num_actions, train_time, test_time, \
    max_time, seed, grader):
    """Test StrategyLearner."""
    # Initialize points for this test case
    points_earned = 0.0
    try:
        incorrect = True
        if not "StrategyLearner" in globals():
            import importlib
            m = importlib.import_module("StrategyLearner")
            globals()["StrategyLearner"] = m
        outsample_cr_to_beat = None
        if benchmark_type == "clean":
            outsample_cr_to_beat = benchmark
        def timeoutwrapper_strategylearner():
            # Set fixed seed for repetability
            np.random.seed(seed)
            random.seed(seed)
            learner = StrategyLearner.StrategyLearner(verbose=False,
                                                      impact=impact,
                                                      num_states=num_states,
                                                      num_actions=num_actions)
            tmp = time.time()
            learner.add_evidence(**insample_args)
            train_t = time.time()-tmp
            tmp = time.time()
            insample_trades_1 = learner.test_policy(**insample_args)
            test_t = time.time()-tmp
            insample_trades_2 = learner.test_policy(**insample_args)
            tmp = time.time()
            outsample_trades = learner.test_policy(**outsample_args)
            out_test_t = time.time()-tmp
            return insample_trades_1, insample_trades_2, outsample_trades, \
                   train_t, test_t, out_test_t
        msgs = []
        in_trades_1, in_trades_2, out_trades, train_t, test_t, out_test_t = \
                run_with_timeout(timeoutwrapper_strategylearner,max_time,(),{})
        incorrect = False
        if len(in_trades_1.shape)!= 2 or in_trades_1.shape[1]!= 1:
            incorrect = True
            msgs.append("  First insample trades DF has invalid shape: {}".
                        format(in_trades_1.shape))
        elif len(in_trades_2.shape)!= 2 or in_trades_2.shape[1]!= 1:
            incorrect = True
            msgs.append("  Second insample trades DF has invalid shape: {}".
                        format(in_trades_2.shape))
        elif len(out_trades.shape)!=2 or out_trades.shape[1]!= 1:
            incorrect = True
            msgs.append("  Out-of-sample trades DF has invalid shape: {}".
                        format(out_trades.shape))
        else:
            tmp_csum = 0.0
            for date,trade in in_trades_1.iterrows():
                tmp_csum += trade.iloc[0]
                if (trade.iloc[0] != 0) and \
                   (trade.abs().iloc[0]!= MAX_HOLDINGS) and \
                   (trade.abs().iloc[0]!= 2 * MAX_HOLDINGS):
                   incorrect = True
                   msgs.append("  Illegal trade in first insample DF. \
                    abs(trade) not one of ({},{},{}).\n  Date {}, Trade {}".
                    format(0, MAX_HOLDINGS, 2 * MAX_HOLDINGS, date, trade))
                   break
                elif abs(tmp_csum) > MAX_HOLDINGS:
                    incorrect = True
                    msgs.append("  Holdings more than {} long or short in \
                        first insample DF. Date {}, Trade {}".
                        format(MAX_HOLDINGS,date,trade))
                    break
            tmp_csum = 0.0
            for date,trade in in_trades_2.iterrows():
                tmp_csum += trade.iloc[0]
                if (trade.iloc[0] != 0) and \
                   (trade.abs().iloc[0]!= MAX_HOLDINGS) and \
                   (trade.abs().iloc[0]!= 2 * MAX_HOLDINGS):
                   incorrect = True
                   msgs.append("  illegal trade in second insample DF. \
                    abs(trade) not one of ({},{},{}).\n  Date {}, Trade {}".
                    format(0, MAX_HOLDINGS, 2 * MAX_HOLDINGS, date, trade))
                   break
                elif abs(tmp_csum) > MAX_HOLDINGS:
                    incorrect = True
                    msgs.append("  holdings more than {} long or short in \
                        second insample DF. Date {}, Trade {}".
                        format(MAX_HOLDINGS, date, trade))
                    break
            tmp_csum = 0.0
            for date,trade in out_trades.iterrows():
                tmp_csum += trade.iloc[0]
                if (trade.iloc[0] != 0) and \
                   (trade.abs().iloc[0] != MAX_HOLDINGS) and \
                   (trade.abs().iloc[0] != 2 * MAX_HOLDINGS):
                   incorrect = True
                   msgs.append("  illegal trade in out-of-sample DF. \
                    abs(trade) not one of ({},{},{}).\n  Date {}, Trade {}".
                    format(0, MAX_HOLDINGS, 2 * MAX_HOLDINGS, date, trade))
                   break
                elif abs(tmp_csum) > MAX_HOLDINGS:
                    incorrect = True
                    msgs.append("  holdings more than {} long or short in \
                        out-of-sample DF. Date {}, Trade {}".format(MAX_HOLDINGS,date,trade))
                    break
        if not(incorrect):
            if train_t > train_time:
                incorrect = True
                msgs.append("  add_evidence() took {} seconds, max allowed {}".
                    format(train_t,train_time))
            else:
                points_earned += 1.0
            if test_t > test_time:
                incorrect = True
                msgs.append("  test_policy() took {} seconds, max allowed {}".
                    format(test_t,test_time))
            else:
                points_earned += 2.0
            if not((in_trades_1 == in_trades_2).all()[0]):
                incorrect = True
                mismatches = in_trades_1.join(in_trades_2,how="outer",lsuffix="1",rsuffix="2")
                mismatches = mismatches[mismatches.ix[:,0]!=mismatches.ix[:,1]]
                msgs.append("  consecutive calls to test_policy() with same \
                    input did not produce same output:")
                msgs.append("  Mismatched trades:\n {}".format(mismatches))
            else:
                points_earned += 2.0
            insample_cr = evaluate_policy(insample_args["symbol"],in_trades_1,
                                          insample_args["start_val"],
                                          insample_args["start_date"],
                                          insample_args["end_date"],
                                          market_impact=impact,
                                          commission_cost=0.0)
            outsample_cr = evaluate_policy(outsample_args["symbol"],out_trades,
                                           outsample_args["start_val"],
                                           outsample_args["start_date"],
                                           outsample_args["end_date"],
                                           market_impact=impact,
                                           commission_cost=0.0)
            if insample_cr <= benchmark:
                incorrect = True
                msgs.append("  in-sample return ({}) did not beat benchmark ({})".
                            format(insample_cr,benchmark))
            else:
                points_earned += 5.0
            if outsample_cr_to_beat is None:
                if out_test_t > test_time:
                    incorrect = True
                    msgs.append("  out-sample took {} seconds, max of {}".
                                format(out_test_t,test_time))
                else:
                    points_earned += 5.0
            else:
                if outsample_cr < outsample_cr_to_beat:
                    incorrect = True
                    msgs.append("  out-sample return ({}) did not beat benchmark ({})".
                                format(outsample_cr,outsample_cr_to_beat))
                else:
                    points_earned += 5.0
        if incorrect:
            inputs_str = "    insample_args: {}\n" \
                         "    outsample_args: {}\n" \
                         "    benchmark_type: {}\n" \
                         "    benchmark: {}\n" \
                         "    train_time: {}\n" \
                         "    test_time: {}\n" \
                         "    max_time: {}\n" \
                         "    seed: {}\n".format(insample_args, outsample_args,
                            benchmark_type, benchmark, train_time, test_time,
                            max_time,seed)
            raise IncorrectOutput("Test failed on one or more output criteria. \
                                  \n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, 
                                  "\n".join(msgs)))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in range(len(tb_list)):
            row = tb_list[i]
            # Show only filename instead of long absolute path
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])
        if tb_list:
            msg += "Traceback:\n"
            msg += "".join(tb.format_list(tb_list))
        elif "grading_traceback" in dir(e):
            msg += "Traceback:\n"
            msg += "".join(tb.format_list(e.grading_traceback))
        msg += "{}: {}".format(e.__class__.__name__, str(e))

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome="failed", points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome="passed", points=points_earned, msg=None))

def evaluate_policy(symbol, trades, startval, start_date, end_date, 
                    market_impact, commission_cost):
    """Compute the cumulative return for a portfolio."""
    orders_df = pd.DataFrame(columns=["Shares","Order","Symbol"])
    for row_idx in trades.index:
        nshares = trades.loc[row_idx][0]
        if nshares == 0:
            continue
        order = "sell" if nshares < 0 else "buy"
        new_row = pd.DataFrame([[abs(nshares),order,symbol],],
                               columns=["Shares","Order","Symbol"],index=[row_idx,])
        orders_df = orders_df.append(new_row)
    portvals = compute_portvals(orders_df, start_date, end_date, startval,
                                market_impact,commission_cost)
    return float(portvals[-1]/portvals[0])-1

def compute_portvals(orders_df, start_date, end_date, startval, 
                     market_impact=0.0, commission_cost=0.0):
    """Simulate the market for the given date range and orders file."""
    symbols = []
    orders = []
    orders_df = orders_df.sort_index()
    for date, order in orders_df.iterrows():
        shares = order["Shares"]
        action = order["Order"]
        symbol = order["Symbol"]
        if action.lower() == "sell":
            shares *= -1
        order = (date, symbol, shares)
        orders.append(order)
        symbols.append(symbol)
    symbols = list(set(symbols))
    dates = pd.date_range(start_date, end_date)
    prices_all = util.get_data(symbols, dates)
    prices = prices_all[symbols]
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    prices["_CASH"] = 1.0
    trades = pd.DataFrame(index=prices.index, columns=symbols)
    trades = trades.fillna(0)
    cash = pd.Series(index=prices.index)
    cash = cash.fillna(0)
    cash.ix[0] = startval
    for date, symbol, shares in orders:
        price = prices[symbol][date]
        val = shares * price
        # Transaction cost model
        val += commission_cost + (pd.np.abs(shares) * price * market_impact)
        positions = prices.ix[date] * trades.sum()
        totalcash = cash.sum()
        if (date < prices.index.min()) or (date > prices.index.max()):
            continue
        trades[symbol][date] += shares
        cash[date] -= val
    trades["_CASH"] = cash
    holdings = trades.cumsum()
    df_portvals = (prices * holdings).sum(axis=1)
    return df_portvals

if __name__ == "__main__":
    pytest.main(["-s", __file__])
