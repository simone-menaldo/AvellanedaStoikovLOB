import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import chi2
import statsmodels.api as sm
from decimal import Decimal
from tqdm import tqdm
import os

os.chdir(r"C:\Users\Lenovo\Documents\Quantitative Finance\Thesis")

# Plot parameters
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Parameters of the strategy
gamma = 0.001
Q     = 30
q0    = 0
T     = 60
T_end = 600
alpha = 0.05

est_days_list = np.arange(1, 20)

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

def median_rv_est(log_rets):

    from statistics import median
    import math

    med_sq_sum = 0.0
    N = len(log_rets)

    # Convert the input to a series
    log_rets = pd.Series(log_rets, index=np.arange(len(log_rets)))

    # Define bakward and forward returns
    log_rets_bwd = log_rets.shift(1)
    log_rets_fwd = log_rets.shift(-1)

    # Subset the variables
    log_rets     = list(log_rets[1:-1])
    log_rets_bwd = list(log_rets_bwd[1:])
    log_rets_fwd = list(log_rets_fwd[:-1])

    for i in range(N - 2):
        med_sq_sum += median([abs(log_rets[i]), abs(log_rets_bwd[i]), abs(log_rets_fwd[i])])**2

    return (math.pi / (6 - 4 * np.sqrt(3) + math.pi) * N / (N - 2) * med_sq_sum) / N

# Variables init
sigma_list = []
A_list     = []
k_list     = []

sigma_up_list = []
A_up_list     = []
k_up_list     = []

sigma_low_list = []
A_low_list     = []
k_low_list     = []

# Variables for plot formatting
axis_label = []

table = open('Code/Results/backtest_sens.txt', 'w')

table.write('Estimation period & N. trades & Mean inv & Std dev inv & Mean P\&L & Std dev P\&L \\\ \n')
table.write('\hline \n')

for est_days in est_days_list:

    print(f'Estimation days: {est_days}')

    '''
    Parameters estimation
    '''

    # Default parameters
    n_est     = 5
    tick_mult = 5
    start_ts  = 37800
    end_ts    = 55800
    tick_size = 0.01

    # Define the bid and ask quotes for the estimation
    est_points = np.arange(1, n_est + 1)
    quotes     = est_points * tick_size * tick_mult

    # Initialize start times and number of intervals
    start_time = np.arange(start_ts, end_ts, T)

    # Variables init
    hit_times = {}
    for quote in quotes:
        hit_times[quote] = []

    mid_prices = []

    # Define the range of days for the estimation
    days_range = ['02', '05', '06', '07', '08', '09', '12', '13', '14', '15', '16', \
         '20', '21', '22', '23', '26', '27', '28', '29', '30']

    days_list = []

    # Format plots x axis
    axis_label.append(f'2015-01-{days_range[est_days]}')

    for d in days_range[:est_days]:

        # Import order book and order flow data
        try:
            order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_message_10.pkl')
            order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_orderbook_10.pkl')
        except:
            continue

        print(f'2015-01-{d}')
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

        # Estimate realized variance
        mid_prices += list(mid_price)

        if d == days_range[est_days - 1]:
            sigma     = np.sqrt(median_rv_est(np.diff(mid_prices)))
            sigma_up  = np.sqrt(((len(mid_prices) - 1) * sigma**2) / chi2.ppf(1 - alpha / 2, df=(len(mid_prices) - 1)))
            sigma_low = np.sqrt(((len(mid_prices) - 1) * sigma**2) / chi2.ppf(alpha / 2, df=(len(mid_prices) - 1)))

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

    # Estimate lambda
    lambda_est = pd.Series(0.0, index=quotes)

    for quote in quotes:
        lambda_est[quote] = 1 / np.mean(hit_times[quote])

    # Estimate k and A through linear regression
    X = sm.add_constant(est_points)
    Y = np.log(lambda_est)

    reg = sm.OLS(Y, X).fit()

    A = np.exp(reg.params[0])
    k = - reg.params[1]

    # Compute the confidence intervals
    conf_int = np.array(reg.conf_int(alpha=alpha))

    A_low = np.exp(conf_int[0, 0])
    A_up  = np.exp(conf_int[0, 1])
    k_low = - conf_int[1, 1]
    k_up  = - conf_int[1, 0]

    # Save the estimated parameters
    sigma_list.append(sigma)
    A_list.append(A)
    k_list.append(k)

    sigma_up_list.append(sigma_up)
    A_up_list.append(A_up)
    k_up_list.append(k_up)
    
    sigma_low_list.append(sigma_low)
    A_low_list.append(A_low)
    k_low_list.append(k_low)

    '''
    Optimal quotes
    '''

    # Variables holding prices and quotes
    trade_prices = []
    bid_prices   = []
    ask_prices   = []
    ref_prices   = []

    trade_count_tot = 0

    # Variables holding inventory and P&L
    inventory = [q0]
    pnl       = [0]

    tick_mult = 10

    # Import order book and order flow data
    try:
        order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-30_34200000_57600000_message_10.pkl')
        order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-30_34200000_57600000_orderbook_10.pkl')
    except:
        continue

    print('2015-01-30')

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

    trade_count = 0

    # Asymptotic bid and ask spreads
    bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
    ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

    bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
    ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

    # Set the initial bid and ask prices
    ref_price = mid_price_df['price'][0]
    ref_time  = mid_price_df['time'][0]

    ref_prices.append(mid_price_df['price'][0])

    bid_price = ref_price - bid_spread_aysm
    ask_price = ref_price + ask_spread_aysm

    # Check if the quotes are hit
    for i in tqdm(range(trades.shape[0])):

        trade_price = trades['price'][i]
        trade_time  = trades['time'][i]
        trade_dir   = trades['dir'][i]

        if (trade_time - ref_time) >= T:

            ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
            ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

            bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
            ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

            bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
            ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

            bid_price = ref_price - bid_spread_aysm
            ask_price = ref_price + ask_spread_aysm       

        if inventory[-1] != Q and trade_dir == 1 and trade_price <= bid_price:

            trade_count += 1

            inventory.append(inventory[-1] + 1)
            pnl.append(pnl[-1] - bid_price)

            ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
            ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

            ref_prices.append(ref_price)

            bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
            ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

            bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
            ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

            bid_price = ref_price - bid_spread_aysm
            ask_price = ref_price + ask_spread_aysm           

        elif inventory[-1] != -Q and trade_dir == -1 and trade_price >= ask_price:

            trade_count += 1

            inventory.append(inventory[-1] - 1)
            pnl.append(pnl[-1] + ask_price)

            ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
            ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

            ref_prices.append(ref_price)

            bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
            ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

            bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
            ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

            bid_price = ref_price - bid_spread_aysm
            ask_price = ref_price + ask_spread_aysm

        else:

            ref_prices.append(ref_price)
            inventory.append(inventory[-1])
            pnl.append(pnl[-1])

        trade_prices.append(trade_price)
        bid_prices.append(bid_price)
        ask_prices.append(ask_price)

    # Update the inventory with the dynamics of the position
    ref_prices_arr = np.array(ref_prices)
    inventory_arr  = np.array(inventory)
    pnl_arr        = np.array(pnl)

    pnl_arr += inventory_arr * ref_prices_arr

    # Generate the table with the summary statistics for each day
    mean_inv = round(np.mean(inventory_arr[-trades.shape[0]:]), 2)
    std_inv  = round(np.std(inventory_arr[-trades.shape[0]:]), 2)
    mean_pnl = round(np.mean(pnl_arr[-trades.shape[0]:]), 2)
    std_pnl  = round(np.std(pnl_arr[-trades.shape[0]:]), 2)

    table.write(f'2015-01-{days_list[-1]} & {trade_count} & {mean_inv} & {std_inv} & {mean_pnl} & {std_pnl} \\\ \n')

# Plot the estimated parameters
fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(sigma_list, color='firebrick')
for i in range(len(sigma_list)):
    plt.vlines(x=i, ymin=sigma_list[i], ymax=sigma_up_list[i], color='blue')
    plt.hlines(y=sigma_up_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
    plt.vlines(x=i, ymin=sigma_low_list[i], ymax=sigma_list[i], color='blue')
    plt.hlines(y=sigma_low_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
plt.xticks(np.arange(len(sigma_list)), axis_label, rotation=45)
plt.xlabel('End date')
plt.ylabel('\u03c3')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/backtest_sens_sigma.png', bbox_inches='tight', dpi=100, format='png')

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(A_list, color='firebrick')
for i in range(len(A_list)):
    plt.vlines(x=i, ymin=A_list[i], ymax=A_up_list[i], color='blue')
    plt.hlines(y=A_up_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
    plt.vlines(x=i, ymin=A_low_list[i], ymax=A_list[i], color='blue')
    plt.hlines(y=A_low_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
plt.xticks(np.arange(len(A_list)), axis_label, rotation=45)
plt.xlabel('End date')
plt.ylabel('A')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/backtest_sens_A.png', bbox_inches='tight', dpi=100, format='png')

fig = plt.figure()
ax  = fig.add_subplot(1, 1, 1)
plt.plot(k_list, color='firebrick')
for i in range(len(k_list)):
    plt.vlines(x=i, ymin=k_list[i], ymax=k_up_list[i], color='blue')
    plt.hlines(y=k_up_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
    plt.vlines(x=i, ymin=k_low_list[i], ymax=k_list[i], color='blue')
    plt.hlines(y=k_low_list[i], xmin=(i - 0.1), xmax=(i + 0.1), color='blue')
plt.xticks(np.arange(len(k_list)), axis_label, rotation=45)
plt.xlabel('End date')
plt.ylabel('\u03ba')
plt.grid()
ax.set_facecolor('#f2f2f2')
plt.savefig(f'Plots/backtest_sens_k.png', bbox_inches='tight', dpi=100, format='png')

table.close()