import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from itertools import combinations
import statsmodels.api as sm
from decimal import Decimal
from tqdm import tqdm
import time
import os

os.chdir(r"C:\Users\Lenovo\Documents\Quantitative Finance\Thesis")

# Plot parameters
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Parameters of the strategy
gamma     = 0.001
end_strat = 0
symmetry  = 1
Q         = 30
q0        = 0
est_days  = 11
T         = 60
T_end     = 600

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

trace = open('Code/Traces/backtest_m3.txt', 'w')

def main(argv):

    global gamma, end_strat, symmetry, Q, q0, est_days, T

    try:
        opts, args = getopt.getopt(argv, 'hg:e:s:Q:q:d:T:')
    except getopt.GetoptError:
        print('backtest_m1.py -g -e -s -Q -q -d -T')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Parameters for the backtest:')
            print('-g: risk adversion')
            print('-e: set end time at each day for the strategy')
            print('-s: estimate equal A and k at both sides')
            print('-Q: inventory limits')
            print('-q: initial inventory')
            print('-d: number of days for the estimation')
            print('-T: time window for quotes update')
            sys.exit()
        elif opt == '-g':
            gamma = float(arg)
        elif opt == '-e':
            end_strat = int(arg)
        elif opt == '-s':
            symmetry = int(arg)
        elif opt == '-Q':
            Q = int(arg)
        elif opt == '-q':
            q0 = int(arg)
        elif opt == '-d':
            est_days = int(arg)
        elif opt == '-T':
            T = int(arg)

    trace.write(f'Parameters for the backtest: gamma={gamma}, end_strat={end_strat}, symmetry={symmetry}, Q={Q}, q0={q0}, est_days={est_days}, T={T}\n')
    trace.write('-' * 100 + '\n')

    backtest_start = time.perf_counter()

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
    date_range = pd.bdate_range('2015-01-01', '2015-01-31')
    days_range = [str(x)[8:10] for x in date_range]

    days_list = []

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
            sigma = np.sqrt(median_rv_est(np.diff(mid_prices)))

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

    trace.write('Estimated parameters:\n')
    trace.write(f'sigma: {sigma}\n')
    trace.write(f'A: {A}\n')
    trace.write(f'k: {k}\n')
    trace.write('-' * 100 + '\n')

    '''
    Optimal quotes
    '''

    table = open('Code/Results/backtest_table_m3.txt', 'w')

    table.write('Day & N. trades & Mean inv & Std dev inv & Mean P\&L & Std dev P\&L \\\ \n')
    table.write('\hline \n')

    # Variables holding prices and quotes
    trade_prices = []
    bid_prices   = []
    ask_prices   = []
    ref_prices   = []

    trade_count_tot = 0

    # Variables holding inventory and P&L
    inventory = [q0]
    pnl       = [0]

    # Variables for plot formatting
    axis_ticks = []
    axis_label = []

    tick_mult = 10

    for d in days_range[est_days:]:

        # Import order book and order flow data
        try:
            order_flow = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_message_10.pkl')
            order_book = pd.read_pickle(f'Code/data/pickle/TSLA/TSLA_2015-01-{d}_34200000_57600000_orderbook_10.pkl')
        except:
            continue

        print(f'2015-01-{d}')

        # Format plots x axis
        axis_ticks.append(len(inventory))
        axis_label.append(f'2015-01-{d}')

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

        # Select between asymptotic quotes and exact quotes
        if end_strat == 0 and symmetry == 1:

            # Asymptotic bid and ask spreads
            bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
            ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

            bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
            ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

            trace.write(f'Bid spread: {bid_spread_aysm}\n')
            trace.write(f'Ask spread: {ask_spread_aysm}\n')
            trace.write('-' * 100 + '\n')

            # Set the initial bid and ask prices
            ref_price = mid_price_df['price'][0]
            ref_time  = mid_price_df['time'][0]

            if d == days_range[est_days]:
                ref_prices.append(mid_price_df['price'][0])

            bid_price = ref_price - bid_spread_aysm
            ask_price = ref_price + ask_spread_aysm

            trace.write(f'Set bid price equal to {bid_price}\n')
            trace.write(f'Set ask price equal to {ask_price}\n')
            trace.write('-' * 100 + '\n')

            # Check if the quotes are hit
            for i in tqdm(range(trades.shape[0])):

                trade_price = trades['price'][i]
                trade_time  = trades['time'][i]
                trade_dir   = trades['dir'][i]

                trace.write(f'[{trade_time}] Trade price equal to {trade_price} with dir {trade_dir}\n')
                trace.write('-' * 100 + '\n')

                if (trade_time - ref_time) >= T:

                    trace.write('Trade window closed, resetting quotes\n')

                    ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
                    ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

                    bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + (2 * inventory[-1] + 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
                    ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) - (2 * inventory[-1] - 1) / 2 * np.sqrt((sigma**2 * gamma) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

                    bid_spread_aysm = tick_round(bid_spread_aysm * tick_size * tick_mult, tick_size)
                    ask_spread_aysm = tick_round(ask_spread_aysm * tick_size * tick_mult, tick_size)

                    trace.write(f'Bid spread: {bid_spread_aysm}\n')
                    trace.write(f'Ask spread: {ask_spread_aysm}\n')
                    trace.write('-' * 100 + '\n')

                    bid_price = ref_price - bid_spread_aysm
                    ask_price = ref_price + ask_spread_aysm

                    trace.write(f'Set bid price equal to {bid_price}\n')
                    trace.write(f'Set ask price equal to {ask_price}\n')
                    trace.write('-' * 100 + '\n')         

                if inventory[-1] != Q and trade_dir == 1 and trade_price <= bid_price:

                    trace.write('Bid quote hit!\n')

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

                    trace.write(f'Bid spread: {bid_spread_aysm}\n')
                    trace.write(f'Ask spread: {ask_spread_aysm}\n')
                    trace.write('-' * 100 + '\n')

                    bid_price = ref_price - bid_spread_aysm
                    ask_price = ref_price + ask_spread_aysm   

                    trace.write(f'Set bid price equal to {bid_price}\n')
                    trace.write(f'Set ask price equal to {ask_price}\n')
                    trace.write('-' * 100 + '\n')                 

                elif inventory[-1] != -Q and trade_dir == -1 and trade_price >= ask_price:

                    trace.write('Ask quote hit!\n')

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

                    trace.write(f'Bid spread: {bid_spread_aysm}\n')
                    trace.write(f'Ask spread: {ask_spread_aysm}\n')
                    trace.write('-' * 100 + '\n')

                    bid_price = ref_price - bid_spread_aysm
                    ask_price = ref_price + ask_spread_aysm

                    trace.write(f'Set bid price equal to {bid_price}\n')
                    trace.write(f'Set ask price equal to {ask_price}\n')
                    trace.write('-' * 100 + '\n')  

                else:

                    ref_prices.append(ref_price)
                    inventory.append(inventory[-1])
                    pnl.append(pnl[-1])

                trade_prices.append(trade_price)
                bid_prices.append(bid_price)
                ask_prices.append(ask_price)

        elif end_strat == 1 and symmetry == 1:

            # Computation of alpha, eta and beta
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of the matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Computation of the v(t) function
            v = pd.Series(expm(-M * T_end) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])

            # Bid and ask spreads
            if inventory[-1] != Q:
                bid_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] + 1}']) + 1 / gamma * np.log(1 + gamma / k)
            if inventory[-1] != -Q:
                ask_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] - 1}']) + 1 / gamma * np.log(1 + gamma / k)

            bid_spread = tick_round(bid_spread * tick_size * tick_mult, tick_size)
            ask_spread = tick_round(ask_spread * tick_size * tick_mult, tick_size)

            # Set the initial bid and ask prices
            ref_price = mid_price_df['price'][0]
            ref_time  = mid_price_df['time'][0]

            bid_price = ref_price - bid_spread
            ask_price = ref_price + ask_spread

            time_mult  = 0
            strat_time = start_ts + T_end * time_mult

            # Check if the quotes are hit
            for i in tqdm(range(trades.shape[0])):

                trade_price = trades['price'][i]
                trade_time  = trades['time'][i]
                trade_dir   = trades['dir'][i]

                if (trade_time - strat_time) >= T_end:

                    time_mult += 1
                    strat_time = start_ts + T_end * time_mult

                if (trade_time - ref_time) >= T:

                    ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
                    ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

                    v = pd.Series(expm(-M * (T_end - (trade_time - strat_time))) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
                    
                    if inventory[-1] != Q:
                        bid_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] + 1}']) + 1 / gamma * np.log(1 + gamma / k)
                    if inventory[-1] != -Q:
                        ask_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] - 1}']) + 1 / gamma * np.log(1 + gamma / k)

                    bid_spread = tick_round(bid_spread * tick_size * tick_mult, tick_size)
                    ask_spread = tick_round(ask_spread * tick_size * tick_mult, tick_size)

                    bid_price = ref_price - bid_spread
                    ask_price = ref_price + ask_spread                    

                if inventory[-1] != Q and trade_dir == 1 and trade_price <= bid_price:

                    inventory.append(inventory[-1] + 1)
                    pnl.append(pnl[-1] - bid_price)

                    ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
                    ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

                    v = pd.Series(expm(-M * (T_end - (trade_time - strat_time))) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
                    
                    if inventory[-1] != Q:
                        bid_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] + 1}']) + 1 / gamma * np.log(1 + gamma / k)
                    if inventory[-1] != -Q:
                        ask_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] - 1}']) + 1 / gamma * np.log(1 + gamma / k)

                    bid_spread = tick_round(bid_spread * tick_size * tick_mult, tick_size)
                    ask_spread = tick_round(ask_spread * tick_size * tick_mult, tick_size)

                    bid_price = ref_price - bid_spread
                    ask_price = ref_price + ask_spread

                elif inventory[-1] != -Q and trade_dir == -1 and trade_price >= ask_price:

                    inventory.append(inventory[-1] - 1)
                    pnl.append(pnl[-1] + ask_price)

                    ref_price = mid_price_df.query(f'time == {trade_time}')['price'].values[0]
                    ref_time  = mid_price_df.query(f'time == {trade_time}')['time'].values[0]

                    v = pd.Series(expm(-M * (T_end - (trade_time - strat_time))) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
                    
                    if inventory[-1] != Q:
                        bid_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] + 1}']) + 1 / gamma * np.log(1 + gamma / k)
                    if inventory[-1] != -Q:
                        ask_spread = 1 / k * np.log(v[f'{inventory[-1]}'] / v[f'{inventory[-1] - 1}']) + 1 / gamma * np.log(1 + gamma / k)

                    bid_spread = tick_round(bid_spread * tick_size * tick_mult, tick_size)
                    ask_spread = tick_round(ask_spread * tick_size * tick_mult, tick_size)

                    bid_price = ref_price - bid_spread
                    ask_price = ref_price + ask_spread

                else:

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

        table.write(f'2015-01-{d} & {trade_count} & {mean_inv} & {std_inv} & {mean_pnl} & {std_pnl} \\\ \n')

        trade_count_tot += trade_count

    # Generate the table with the summary statistics for the period
    mean_inv = round(np.mean(inventory_arr), 2)
    std_inv  = round(np.std(inventory_arr), 2)
    mean_pnl = round(np.mean(pnl_arr), 2)
    std_pnl  = round(np.std(pnl_arr), 2)

    table.write('\hline \n')
    table.write(f'Total & {trade_count_tot} & {mean_inv} & {std_inv} & {mean_pnl} & {std_pnl} \\\ \n')
    table.write('\hline \n')

    # Set the initial parameters as booleans
    if end_strat == 0:
        end_strat = False
    else:
        end_strat = True

    if symmetry == 0:
        symmetry = False
    else:
        symmetry = True

    # Plot the results
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(inventory_arr, color='blue')
    plt.axhline(Q, linestyle='--', color='firebrick')
    plt.axhline(-Q, linestyle='--', color='firebrick')
    plt.xticks(axis_ticks, axis_label, rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Inventory')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    plt.savefig(f'Plots/backtest_m3_inv_{end_strat}_{symmetry}.png', bbox_inches='tight', dpi=100, format='png')

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(pnl_arr, color='firebrick')
    plt.xticks(axis_ticks, axis_label, rotation=45)
    plt.xlabel('Date')
    plt.ylabel('P&L')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    plt.savefig(f'Plots/backtest_m3_pnl_{end_strat}_{symmetry}.png', bbox_inches='tight', dpi=100, format='png')

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(trade_prices, color='blue')
    plt.plot(bid_prices, color='green')
    plt.plot(ask_prices, color='red')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    plt.savefig(f'Plots/backtest_m3_prices_{end_strat}_{symmetry}.png', bbox_inches='tight', dpi=100, format='png')

    trace.close()
    table.close()

    print(f'Total time elapsed for the backtest: {time.perf_counter() - backtest_start}')

if __name__ == '__main__':
    main(sys.argv[1:])