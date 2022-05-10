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
plt.rcParams['font.size'] = 18
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

# Open a file for writing the results
res_sum = open('Code/Results/param_est_m3.txt', 'w')
res_sum.write('Estimation of the parameters A and K with estimator 1 and equal trade price\n')
res_sum.write('-' * 100 + '\n')

# Default parameters
n_est     = 5
T         = 60
tick_mult = 5
start_ts  = 37800
end_ts    = 55800
tick_size = 0.01

# Define the bid and ask quotes for the estimation
est_points = np.arange(1, n_est + 1)
quotes     = est_points * tick_size * tick_mult
points = pd.Series(est_points, index=quotes)


# Initialize start times and number of intervals
start_time = np.arange(start_ts, end_ts, T)
n_ints     = int((end_ts - start_ts) / T)

# Variables init
hit_times = {}
for quote in quotes:
    hit_times[quote] = []

# Define the range of days for the estimation
date_range = pd.bdate_range('2015-01-01', '2015-01-31')
days_range = [str(x)[8:10] for x in date_range]

for d in days_range:

    # Import order book and order flow data
    try:
        order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_message_10.pkl')
        order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_orderbook_10.pkl')
        print(f'2015-01-{d}')
    except:
        continue

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

    # Estimate the probability that one of the quotes is hit
    for quote in tqdm(quotes):
        hit_times_day = []
        ref_price  = mid_price_df['price'][0]
        start_time = mid_price_df['time'][0]
        for i in range(trades.shape[0]):
            trade_price = trades['price'][i]
            trade_time  = trades['time'][i]
            trade_dir   = trades['dir'][i]
            if (trade_dir == -1 and trade_price <= (ref_price - quote)) or (trade_dir == 1 and trade_price >= (ref_price + quote)):
                hit_times_day.append(trade_time - start_time)
                ref_price  = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
                start_time = mid_price_df.query(f'time == {trade_time}')['time'].values[0]
        hit_times[quote] += hit_times_day

# # Estimate lambda
lambda_est = pd.Series(0.0, index=quotes)

for quote in quotes:
    lambda_est[quote] = 1 / np.mean(hit_times[quote])

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_est, linestyle='--', marker='o', color='firebrick')
plt.xticks(quotes, labels=(quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the bid and ask sides for TSLA'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_est_m3.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

# Estimate k and A through problem inversion for the bid side
A_bid      = []
k_bid      = []
labels_bid = []

for quote1 in quotes:
    for quote2 in quotes:
        if quote2 > quote1:
            k_bid.append(np.log((lambda_est[quote2]) / (lambda_est[quote1])) / (points[quote1] - points[quote2]))
            A_bid.append((lambda_est[quote1]) * np.exp(k_bid[-1] * points[quote1]))
            labels_bid.append((quote1, quote2))

lambda_fit = []

for i in range(len(labels_bid)):
    lambda_fit_quote = A_bid[i] * np.exp(-k_bid[i] * points)
    lambda_fit.append(lambda_fit_quote)

# Plot the results
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
for i in range(len(labels_bid)):
    plt.plot(lambda_fit[i], label=labels_bid[i])
plt.xticks(quotes, labels=(quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig('Plots/lambda_fit_m3_inv.png', bbox_inches='tight', dpi=100, format='png')