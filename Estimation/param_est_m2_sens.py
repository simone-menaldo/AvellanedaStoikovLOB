#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from decimal import Decimal
import os

os.chdir(r"C:\Users\Lenovo\Documents\Quantitative Finance\Thesis")

# Plot options
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Functions
def order_book_columns(n_levels):

    side  = ['ask', 'bid']
    data  = ['price', 'size']
    level = list(range(1, n_levels + 1))
    combs = list(combinations(side + data + level, 3))
    sel_combs = [comb for comb in combs if (comb[0] in side and comb[1] in data and comb[2] in level)]
    sel_combs.sort(key=lambda tup: tup[2])
    sel_combs = [(x, y, str(z)) for x, y, z in sel_combs]

    return ['_'.join(comb) for comb in sel_combs]

def tick_round(n, tick):
    decs = abs(Decimal(str(tick)).as_tuple().exponent)
    return (tick * (np.array(n) / tick).round()).round(decs)

def poisson_process_gen(lambda_est, T, start_ts, end_ts):

    events_times = [0.0]

    # Simulate the waiting times
    while events_times[-1] <= (end_ts - start_ts):
        waiting_time = np.random.exponential(scale=(1 / lambda_est))
        events_times.append(events_times[-1] + waiting_time)

    # Subset the event times to the selected interval
    events_times = np.array(events_times[1:]) + start_ts

    # Compute the number of events for each interval
    t0 = np.arange(start_ts, end_ts, T)
    events_int = []

    for t in t0:
        events_int.append(len(events_times[(events_times >= t) & (events_times < (t + T))]))

    return list(events_times), list(events_int)

# Default parameters
n_est     = 5
tick_mult = 5
start_ts  = 37800
end_ts    = 55800
tick_size = 0.01
T_range   = [60, 120, 300, 600, 1800]

# Define the bid and ask quotes for the estimation
est_points = np.arange(1, n_est + 1)

ask_quotes = est_points * tick_size * tick_mult
bid_quotes = est_points * tick_size * tick_mult

# Define the range of days for the estimation
date_range = pd.bdate_range('2015-01-01', '2015-01-31')
days_range = [str(x)[8:10] for x in date_range]

# Initialize lambda for bid and ask
lambda_bid_list = []
lambda_ask_list = []

lambda_bid_poi_list = []
lambda_ask_poi_list = []

for T in T_range:

    # Initialize start times and number of intervals
    start_time = np.arange(start_ts, end_ts, T)

    # Variables init
    bid_hit_n_tot  = pd.Series(0.0, index=bid_quotes)
    bid_hit_wt_tot = pd.Series(0.0, index=bid_quotes)
    ask_hit_n_tot  = pd.Series(0.0, index=ask_quotes)
    ask_hit_wt_tot = pd.Series(0.0, index=ask_quotes)

    day_count = 0
    days_list = []

    for d in days_range:

        # Import order book and order flow data
        try:
            order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_message_10.pkl')
            order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_orderbook_10.pkl')
        except:
            continue

        print(f'2015-01-{d}')

        day_count += 1
        days_list.append(d)

        # Rename the columns
        order_flow.columns = ['time', 'event', 'id', 'size', 'price', 'dir']
        order_book.columns = order_book_columns(n_levels=int(order_book.shape[1] / 4))

        # Rescale the prices columns
        order_flow['price'] /= 10000
        for i in range(int(order_book.shape[1] / 2)):
            order_book.iloc[:, 2 * i] /= 10000

        # Compute the mid-price and approximate according to the tick size
        mid_price = (order_book['ask_price_1'] + order_book['bid_price_1']) / 2
        mid_price = tick_round(mid_price, tick_size)

        # Add the timestamps to the midprice
        mid_price_df = pd.DataFrame(mid_price, columns=['price'])
        mid_price_df['time'] = order_flow['time']

        # Extract the trades from the order flow
        trades = order_flow.query('event == 4').reset_index(drop=True)

        # Estimate the probability that each quote is hit
        bid_hit_n  = pd.Series(0.0, index=bid_quotes)
        bid_hit_wt = pd.Series(0.0, index=bid_quotes)
        ask_hit_n  = pd.Series(0.0, index=ask_quotes)
        ask_hit_wt = pd.Series(0.0, index=ask_quotes)

        for t0 in tqdm(start_time):
            window = trades.query(f'time >= {t0} and time < {t0 + T}').reset_index(drop=True)
            mid_price_df['diff'] = (mid_price_df['time'] - t0).abs()
            ref_price = mid_price_df[mid_price_df['diff'] == min(mid_price_df['diff'])]['price'].values[0]
            for quote in bid_quotes:
                bid_price = ref_price - quote
                ask_price = ref_price + quote
                bid_count = 0
                ask_count = 0
                for i in range(len(window)):
                    trade_price = window['price'][i]
                    trade_time  = window['time'][i]
                    trade_dir   = window['dir'][i]
                    if trade_price <= bid_price and trade_dir == -1:
                        bid_count += 1
                        bid_hit_n[quote]  += 1
                        bid_hit_wt[quote] += trade_time - t0
                        break
                    if trade_price >= ask_price and trade_dir == 1:
                        ask_count += 1
                        ask_hit_n[quote]  += 1
                        ask_hit_wt[quote] += trade_time - t0
                        break
                if bid_count == 0:
                    bid_hit_wt += T
                if ask_count == 0:
                    ask_hit_wt += T

        bid_hit_n_tot  += bid_hit_n
        bid_hit_wt_tot += bid_hit_wt
        ask_hit_n_tot  += ask_hit_n
        ask_hit_wt_tot += ask_hit_wt

    # Estimate lambda for each quote
    lambda_bid = bid_hit_n_tot / bid_hit_wt_tot
    lambda_ask = ask_hit_n_tot / ask_hit_wt_tot

    lambda_bid_list.append(lambda_bid)
    lambda_ask_list.append(lambda_ask)

# Set the lambda lists ad dataframes
lambda_bid_df = pd.DataFrame(lambda_bid_list, index=T_range, columns=bid_quotes)
lambda_ask_df = pd.DataFrame(lambda_ask_list, index=T_range, columns=ask_quotes)

# Plot the results
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for quote in bid_quotes:
    plt.plot(lambda_bid_df[quote], linestyle='--', marker='o', label=f'{quote / tick_size}')
plt.xlabel('\u0394T')
plt.legend(title='\u03b4 (ticks)')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/lambda_bid_sens_m2.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for quote in ask_quotes:
    plt.plot(lambda_ask_df[quote], linestyle='--', marker='o', label=f'{quote / tick_size}')
plt.xlabel('\u0394T')
plt.legend(title='\u03b4 (ticks)')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/lambda_ask_sens_m2.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

'''
Poisson process
'''

for T in T_range:

    start_time = np.arange(start_ts, end_ts, T)

    lambda_bid_poi      = pd.Series(0.0, index=bid_quotes)
    poisson_process_bid = {}
    poisson_t_bid       = {}

    for quote in bid_quotes:
        poisson_wt_bid      = 0.0
        poisson_count_bid   = []
        for d in days_list:
            poisson_t_bid[quote], poisson_process_bid[quote] = poisson_process_gen(lambda_bid_df[quote].iloc[0], T, start_ts, end_ts)
            poisson_t_bid[quote]       = np.array(poisson_t_bid[quote])
            poisson_process_bid[quote] = np.array(poisson_process_bid[quote])
            for t0 in start_time:
                poisson_times_bid = poisson_t_bid[quote][(poisson_t_bid[quote] >= t0) & (poisson_t_bid[quote] < (t0 + T))]
                poisson_wt_bid   += sum(poisson_times_bid - t0)
            poisson_wt_bid += sum(poisson_process_bid[quote] == 0) * T
            poisson_count_bid += list(poisson_process_bid[quote])
        lambda_bid_poi[quote] = sum(np.array(poisson_count_bid) != 0) / poisson_wt_bid

    lambda_ask_poi      = pd.Series(0.0, index=ask_quotes)
    poisson_process_ask = {}
    poisson_t_ask       = {}

    for quote in ask_quotes:
        poisson_wt_ask      = 0.0
        poisson_count_ask   = []
        for d in days_list:
            poisson_t_ask[quote], poisson_process_ask[quote] = poisson_process_gen(lambda_ask_df[quote].iloc[0], T, start_ts, end_ts)
            poisson_t_ask[quote]       = np.array(poisson_t_ask[quote])
            poisson_process_ask[quote] = np.array(poisson_process_ask[quote])
            for t0 in start_time:
                poisson_times_ask = poisson_t_ask[quote][(poisson_t_ask[quote] >= t0) & (poisson_t_ask[quote] < (t0 + T))]
                poisson_wt_ask   += sum(poisson_times_ask - t0)
            poisson_wt_ask += sum(poisson_process_ask[quote] == 0) * T
            poisson_count_ask += list(poisson_process_ask[quote])
        lambda_ask_poi[quote] = sum(np.array(poisson_count_ask) != 0) / poisson_wt_ask

    lambda_bid_poi_list.append(lambda_bid_poi)
    lambda_ask_poi_list.append(lambda_ask_poi)

# Set the lambda lists for the poisson processes as dataframes
lambda_bid_poi_df = pd.DataFrame(lambda_bid_poi_list, index=T_range, columns=bid_quotes)
lambda_ask_poi_df = pd.DataFrame(lambda_ask_poi_list, index=T_range, columns=ask_quotes)

# Plot the results
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for quote in bid_quotes:
    plt.plot(lambda_bid_poi_df[quote], linestyle='--', marker='o', label=f'{round(lambda_bid_df[quote].iloc[0], 4)}')
plt.xlabel('\u0394T')
plt.legend(title='\u03bb', loc='upper right')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/lambda_bid_poi_sens_m2.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for quote in ask_quotes:
    plt.plot(lambda_ask_poi_df[quote], linestyle='--', marker='o', label=f'{round(lambda_ask_df[quote].iloc[0], 4)}')
plt.xlabel('\u0394T')
plt.legend(title='\u03bb', loc='upper right')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/lambda_ask_poi_sens_m2.png', bbox_inches='tight', dpi=100, format='png')
plt.close()