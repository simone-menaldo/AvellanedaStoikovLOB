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

# Default parameters
n_est     = 5
T         = 60
tick_mult = 5
start_ts  = 37800
end_ts    = 55800
tick_size = 0.01

# Define the bid and ask quotes for the estimation
est_points = np.arange(1, n_est + 1)

ask_quotes = est_points * tick_size * tick_mult
bid_quotes = est_points * tick_size * tick_mult

bid_points = pd.Series(est_points, index=bid_quotes)
ask_points = pd.Series(est_points, index=ask_quotes)

# Initialize start times and number of intervals
start_time = np.arange(start_ts, end_ts, T)
n_ints     = int((end_ts - start_ts) / T)

# Variables init
bid_hit_n = {}
for quote in bid_quotes:
    bid_hit_n[quote] = []

ask_hit_n = {}
for quote in ask_quotes:
    ask_hit_n[quote] = []

# Define the range of days for the estimation
date_range = pd.bdate_range('2015-01-01', '2015-01-31')
days_range = [str(x)[8:10] for x in date_range]

day_count  = 0

for d in days_range:

    # Import order book and order flow data
    try:
        order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_message_10.pkl')
        order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_orderbook_10.pkl')
    except:
        continue

    print(f'2015-01-{d}')
    day_count += 1

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
    buy_trades  = order_flow.query('event == 4 and dir == 1').reset_index(drop=True)
    sell_trades = order_flow.query('event == 4 and dir == -1').reset_index(drop=True)

    # Estimate the probability that one of the quotes is hit
    for t0 in tqdm(start_time):
        sell_window = sell_trades.query(f'time >= {t0} and time < {t0 + T}').reset_index(drop=True)
        buy_window  = buy_trades.query(f'time >= {t0} and time < {t0 + T}').reset_index(drop=True)
        mid_price_df['diff'] = (mid_price_df['time'] - t0).abs()
        ref_price = mid_price_df[mid_price_df['diff'] == min(mid_price_df['diff'])]['price'].values[0]
        for quote in bid_quotes:
            bid_price = ref_price - quote
            ask_price = ref_price + quote
            if any(sell_window['price'] <= bid_price):
                bid_hit_n[quote].append(sum(sell_window['price'] <= bid_price))
            if any(buy_window['price'] >= ask_price):
                ask_hit_n[quote].append(sum(buy_window['price'] >= ask_price))

# Estimate lambda for each quote
lambda_bid = pd.Series(0.0, index=bid_quotes)
for quote in bid_quotes:
    lambda_bid[quote] = (sum(bid_hit_n[quote]) / (n_ints * day_count)) / T

lambda_ask = pd.Series(0.0, index=ask_quotes)
for quote in ask_quotes:
    lambda_ask[quote] = (sum(ask_hit_n[quote]) / (n_ints * day_count)) / T

# Estimate k and A through problem inversion for the bid side
A_bid      = []
k_bid      = []
labels_bid = []

for quote1 in bid_quotes:
    for quote2 in bid_quotes:
        if quote2 > quote1:
            k_bid.append(np.log((lambda_bid[quote2]) / (lambda_bid[quote1])) / (bid_points[quote1] - bid_points[quote2]))
            A_bid.append((lambda_bid[quote1]) * np.exp(k_bid[-1] * bid_points[quote1]))
            labels_bid.append((quote1, quote2))

lambda_bid_fit = []

for i in range(len(labels_bid)):
    lambda_bid_fit_quote = A_bid[i] * np.exp(-k_bid[i] * bid_points)
    lambda_bid_fit.append(lambda_bid_fit_quote)

# Estimate k and A through problem inversion for the ask side
A_ask      = []
k_ask      = []
labels_ask = []

for quote1 in ask_quotes:
    for quote2 in ask_quotes:
        if quote2 > quote1:
            k_ask.append(np.log((lambda_ask[quote2]) / (lambda_ask[quote1])) / (ask_points[quote1] - ask_points[quote2]))
            A_ask.append((lambda_ask[quote1]) * np.exp(k_ask[-1] * ask_points[quote1]))
            labels_ask.append((quote1, quote2))

lambda_ask_fit = []

for i in range(len(labels_ask)):
    lambda_ask_fit_quote = A_ask[i] * np.exp(-k_ask[i] * ask_points)
    lambda_ask_fit.append(lambda_ask_fit_quote)

# Plot the results
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for i in range(len(labels_bid)):
    plt.plot(lambda_bid_fit[i], label=labels_bid[i])
plt.xticks(bid_quotes, labels=(bid_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig('Plots/lambda_bid_m1_inv.png', bbox_inches='tight', dpi=100, format='png')

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for i in range(len(labels_ask)):
    plt.plot(lambda_ask_fit[i], label=labels_ask[i])
plt.xticks(ask_quotes, labels=(ask_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig('Plots/lambda_ask_m1_inv.png', bbox_inches='tight', dpi=100, format='png')