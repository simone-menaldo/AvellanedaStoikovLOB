#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from decimal import Decimal
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import time
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

def plot_se(est_points, param, param_err, bound):
    for p in est_points:
        plt.vlines(x=p, ymin=param[p], ymax=(param[p] + 1.96 * param_err[p]))
        plt.hlines(y=(param[p] + 1.96 * param_err[p]), xmin=(p - bound), xmax=(p + bound))
        plt.vlines(x=p, ymin=(param[p] - 1.96 * param_err[p]), ymax=param[p])
        plt.hlines(y=(param[p] - 1.96 * param_err[p]), xmin=(p - bound), xmax=(p + bound))

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

# Open a file for writing the results
res_sum = open('Code/Results/param_est_m1.txt', 'w')
res_sum.write('Estimation of the parameters A and K with estimator 1 and equal trade price\n')
res_sum.write('-' * 100 + '\n')

estimation_start = time.perf_counter()

# Default parameters
n_est     = 5
T         = 60
tick_mult = 5
start_ts  = 37800
end_ts    = 55800
tick_size = 0.01

res_sum.write('Parameters:\n')
res_sum.write(f'Number of estimations: {n_est}\n')
res_sum.write(f'Time window: {T}\n')
res_sum.write(f'Daily range: from {start_ts} to {end_ts}\n')
res_sum.write(f'Tick size: {tick_size}\n')
res_sum.write('-' * 100 + '\n')

# Define the bid and ask quotes for the estimation
est_points = np.arange(1, n_est + 1)

ask_quotes = est_points * tick_size * tick_mult
bid_quotes = est_points * tick_size * tick_mult

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

day_count = 0
days_list = []

'''
Estimation of lambda
'''

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

# Estimate the error for lambda at each quote
lambda_bid_err = pd.Series(0.0, index=bid_quotes)
for quote in bid_quotes:
    lambda_bid_err[quote] = (np.std(bid_hit_n[quote]) / np.sqrt(n_ints * day_count)) / T

lambda_ask_err = pd.Series(0.0, index=ask_quotes)
for quote in ask_quotes:
    lambda_ask_err[quote] = (np.std(ask_hit_n[quote]) / np.sqrt(n_ints * day_count)) / T

# Plot the result of the estimation
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_bid, linestyle='--', marker='o', color='firebrick')
plot_se(bid_quotes, lambda_bid, lambda_bid_err, tick_size / 2)
plt.xticks(bid_quotes, labels=(bid_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the bid and ask sides for TSLA'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_bid_est_m1.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_ask, linestyle='--', marker='o', color='firebrick')
plot_se(ask_quotes, lambda_ask, lambda_ask_err, tick_size / 2)
plt.xticks(ask_quotes, labels=(ask_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the bid and ask sides for TSLA'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_ask_est_m1.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

'''
Estimation of A and k
'''

# Estimate k and A through OLS regression for the bid side
X = sm.add_constant(est_points)
Y = np.log(lambda_bid)

reg_bid = sm.OLS(Y, X).fit()

res_sum.write('Lambda bid OLS regression:\n')
res_sum.write(str(reg_bid.summary()) + '\n')
res_sum.write('-' * 100 + '\n')

A_bid = np.exp(reg_bid.params[0])
k_bid = - reg_bid.params[1]

res_sum.write(f'A (bid): {round(A_bid, 5)}\n')
res_sum.write(f'k (bid): {round(k_bid, 5)}\n')
res_sum.write('-' * 100 + '\n')

lambda_bid_ols = pd.Series(np.exp(np.log(A_bid) - k_bid * est_points), index=bid_quotes)

# Estimate k and A through OLS regression for the ask side
X = sm.add_constant(est_points)
Y = np.log(lambda_ask)

reg_ask = sm.OLS(Y, X).fit()

res_sum.write('Lambda ask OLS regression:\n')
res_sum.write(str(reg_ask.summary()) + '\n')
res_sum.write('-' * 100 + '\n')

A_ask = np.exp(reg_ask.params[0])
k_ask = - reg_ask.params[1]

res_sum.write(f'A (ask): {round(A_ask, 5)}\n')
res_sum.write(f'k (ask): {round(k_ask, 5)}\n')
res_sum.write('-' * 100 + '\n')

lambda_ask_ols = pd.Series(np.exp(np.log(A_ask) - k_ask * est_points), index=ask_quotes)

render = Stargazer([reg_bid, reg_ask])

res_sum.write('Latex render:\n')
res_sum.write(render.render_latex() + '\n')
res_sum.write('-' * 100 + '\n')

# Plot the results of the regression
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_bid, linestyle='--', marker='o', color='firebrick', label='\u03BB estimated')
plt.plot(lambda_bid_ols, linestyle='-', color='blue', label='\u03BB fitted')
plt.xticks(bid_quotes, labels=(bid_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the bid side for TSLA: A={round(A_bid, 4)} and k={round(k_bid, 4)}'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_bid_ols_m1.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_ask, linestyle='--', marker='o', color='firebrick', label='\u03BB estimated')
plt.plot(lambda_ask_ols, linestyle='-', color='blue', label='\u03BB fitted')
plt.xticks(ask_quotes, labels=(ask_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the ask side for TSLA: A={round(A_ask, 4)} and k={round(k_ask, 4)}'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_ask_ols_m1.png', bbox_inches='tight', dpi=100, format='png')
plt.close()

res_sum.close()

print(f'Total time elapsed for the estimation: {time.perf_counter() - estimation_start}')

'''
Poisson process simulation
'''

# Generate a Poisson process for the bid and ask sides
lambda_bid_poi      = pd.Series(0.0, index=bid_quotes)
poisson_process_bid = {}

for quote in bid_quotes:
    poisson_process_bid[quote] = []
    for d in days_list:
        poisson_process_bid[quote] += poisson_process_gen(lambda_bid[quote], T, start_ts, end_ts)[1]
    lambda_bid_poi[quote] = (sum(poisson_process_bid[quote]) / (n_ints * day_count)) / T

lambda_ask_poi      = pd.Series(0.0, index=ask_quotes)
poisson_process_ask = {}

for quote in ask_quotes:
    poisson_process_ask[quote] = []
    for d in days_list:
        poisson_process_ask[quote] += poisson_process_gen(lambda_ask[quote], T, start_ts, end_ts)[1]
    lambda_ask_poi[quote] = (sum(poisson_process_ask[quote]) / (n_ints * day_count)) / T

# Plot the results of the estimation
plt.rcParams['font.size'] = 18
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(lambda_bid_poi, linestyle='--', marker='o', color='green', label='Bid')
plt.plot(lambda_ask_poi, linestyle='--', marker='o', color='firebrick', label='Ask')
plt.xticks(bid_quotes, labels=(bid_quotes / tick_size))
plt.xlabel('\u03b4 (ticks)')
plt.legend()
plt.grid()
ax.set_facecolor('#f2f2f2')
# caption = f'Estimation of \u03bb at the bid and ask sides for TSLA'
# plt.figtext(0.5, 0, caption, ha='center', alpha=0.75)
plt.savefig('Plots/lambda_poi_m1.png', bbox_inches='tight', dpi=100, format='png')
plt.close()