#!/usr/bin/python
import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm
import time
import os

os.chdir(r"C:\Users\Lenovo\Documents\Quantitative Finance\Thesis")

# Plot parameters
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Variables initialization
s0    = 100
T     = 600
mu    = 0.001
xi    = 0.01
sigma = 0.3
q0    = 0
gamma = 0.01
k     = 0.3
A     = 0.5
Q     = 10
n_sim = 1000
seed  = 123

def main(argv):

    # Define start time
    start_time = time.perf_counter()

    # Import global variables
    global s0, T, sigma, q0, gamma, k, A, Q, n_sim, seed

    # Define script parameters
    try:
        opts, args = getopt.getopt(argv, 'hp:T:m:x:s:q:g:k:A:Q:n:d')
    except getopt.GetoptError:
        print('av_model_sim.py -p -T -m -x -s -q -g -k -A -Q -n -d')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Parameters for the model')
            print('p: initial stock price')
            print('T: final time of the simulation')
            print('m: drift of the stock price')
            print('x: market impact')
            print('s: volaility of the stock price')
            print('q: initial inventory')
            print('g: risk adversion')
            print('k: price sensitivity of market participants')
            print('A: order rate arrival multiplier')
            print('Q: maximum inventory')
            print('n: number of simulations')
            print('d: seed for the simulation')
            sys.exit()
        elif opt == '-p':
            s0 = float(arg)
        elif opt == '-T':
            T = int(arg)
        elif opt == '-s':
            sigma = float(arg)
        elif opt == '-q':
            q0 = float(arg)
        elif opt == '-g':
            gamma = float(arg)
        elif opt == '-k':
            k = float(arg)
        elif opt == '-A':
            A = int(arg)
        elif opt == '-Q':
            Q = int(arg)
        elif opt == '-n':
            n_sim = int(arg)
        elif opt == '-d':
            seed = int(arg)

    # Set the seed
    np.random.seed(seed)

    # Variables holding inventory and P&L
    inventory_stand = [q0] * n_sim
    pnl_stand       = [0] * n_sim
    
    inventory_drift = [q0] * n_sim
    pnl_drift       = [0] * n_sim
    
    inventory_impct = [q0] * n_sim
    pnl_impct       = [0] * n_sim

    # Variables holding the price processes
    ask_price_stand = [0] * (T + 1)
    bid_price_stand = [0] * (T + 1)
    mid_price_stand = [0] * (T + 1)

    ask_price_drift = [0] * (T + 1)
    bid_price_drift = [0] * (T + 1)
    mid_price_drift = [0] * (T + 1)

    ask_price_impct = [0] * (T + 1)
    bid_price_impct = [0] * (T + 1)
    mid_price_impct = [0] * (T + 1)

    # Computation of alpha, eta and beta
    alpha = k / 2 * gamma * sigma**2
    eta   = A * (1 + gamma / k)**(-(1 + k / gamma))
    beta  = k * mu

    # Matrix M for standard model
    up_diag_stand  = alpha * (Q - np.arange(0, Q + 1))**2
    low_diag_stand = alpha * (Q - np.arange(0, Q)[::-1])**2
    M_stand        = np.diag(np.concatenate((up_diag_stand, low_diag_stand)))
    for i in range(M_stand.shape[0]):
        for j in range(M_stand.shape[1]):
            if j == (i + 1) or j == (i - 1):
                M_stand[i, j] = -eta

    # Matrix M for drift model
    up_diag_drift  = alpha * (Q - np.arange(0, Q + 1))**2 + beta * (Q - np.arange(0, Q + 1))
    low_diag_drift = alpha * (Q - np.arange(0, Q)[::-1])**2 - beta * (Q - np.arange(0, Q)[::-1])
    M_drift        = np.diag(np.concatenate((up_diag_drift, low_diag_drift)))
    for i in range(M_drift.shape[0]):
        for j in range(M_drift.shape[1]):
            if j == (i + 1) or j == (i - 1):
                M_drift[i, j] = -eta

    # Matrix M for impact model
    up_diag_impct  = alpha * (Q - np.arange(0, Q + 1))**2
    low_diag_impct = alpha * (Q - np.arange(0, Q)[::-1])**2
    M_impct        = np.diag(np.concatenate((up_diag_impct, low_diag_impct)))
    for i in range(M_impct.shape[0]):
        for j in range(M_impct.shape[1]):
            if j == (i + 1) or j == (i - 1):
                M_impct[i, j] = -eta * np.exp(- k / 2 * xi)
    
    # Simulation for the standard model
    print('Standard model:')
    for i in tqdm(range(n_sim)):

        # Stock price for the standard model
        white_noise_stand = sigma * (2 * np.random.binomial(n=1, p=0.5, size=T) - 1)
        stock_price_stand = s0 + np.cumsum(white_noise_stand)
        stock_price_stand = np.insert(stock_price_stand, 0, s0)

        for t, s in enumerate(stock_price_stand):

            # v(t) function
            v_stand = pd.Series(expm(-M_stand * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
            
            # Bid and ask spreads
            try:
                ask_spread_stand = 1 / k * np.log(v_stand[f'{inventory_stand[i]}'] / v_stand[f'{inventory_stand[i] - 1}']) + 1 / gamma * np.log(1 + gamma / k)
            except:
                ask_spread_stand = 0.0
            try:
                bid_spread_stand = 1 / k * np.log(v_stand[f'{inventory_stand[i]}'] / v_stand[f'{inventory_stand[i] + 1}']) + 1 / gamma * np.log(1 + gamma / k)                
            except:
                bid_spread_stand = 0.0

            # Set bid and ask probabiliy
            if ask_spread_stand == 0.0:
                ask_prob_stand = 0.0
                bid_prob_stand = A * np.exp(-k * bid_spread_stand)
            elif bid_spread_stand == 0.0:
                ask_prob_stand = A * np.exp(-k * ask_spread_stand)
                bid_prob_stand = 0.0
            else:
                ask_prob_stand = A * np.exp(-k * ask_spread_stand)
                bid_prob_stand = A * np.exp(-k * bid_spread_stand)
            
            ask_prob_stand = max(0, min(ask_prob_stand, 1))
            bid_prob_stand = max(0, min(bid_prob_stand, 1))

            # Simulate whether a buy or sell order arrives
            ask_action_stand = np.random.binomial(n=1, p=ask_prob_stand)
            bid_action_stand = np.random.binomial(n=1, p=bid_prob_stand)

            # Compute inventory and P&L
            inventory_stand[i] -= ask_action_stand
            pnl_stand[i]       += ask_action_stand * (s + ask_spread_stand)
            inventory_stand[i] += bid_action_stand
            pnl_stand[i]       -= bid_action_stand * (s - bid_spread_stand)

            if i == 0:
                ask_price_stand[t] = s + ask_spread_stand
                bid_price_stand[t] = s - bid_spread_stand
                mid_price_stand[t] = s

        pnl_stand[i] += inventory_stand[i] * s

    # Simulation for the drift model
    print('Drift model:')
    for i in tqdm(range(n_sim)):

        # Stock price for the drift model
        white_noise_drift = mu + sigma * (2 * np.random.binomial(n=1, p=0.5, size=T) - 1)
        stock_price_drift = s0 + np.cumsum(white_noise_drift)
        stock_price_drift = np.insert(stock_price_drift, 0, s0)

        for t, s in enumerate(stock_price_drift):

            # v(t) function
            v_drift = pd.Series(expm(-M_drift * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])

            # Bid and ask spreads
            try:
                ask_spread_drift = 1 / k * np.log(v_drift[f'{inventory_drift[i]}'] / v_drift[f'{inventory_drift[i] - 1}']) + 1 / gamma * np.log(1 + gamma / k)
            except:
                ask_spread_drift = 0.0
            try:
                bid_spread_drift = 1 / k * np.log(v_drift[f'{inventory_drift[i]}'] / v_drift[f'{inventory_drift[i] + 1}']) + 1 / gamma * np.log(1 + gamma / k)
            except:
                bid_spread_drift = 0.0

            # Set bid and ask probabiliy
            if ask_spread_drift == 0.0:
                ask_prob_drift = 0.0
                bid_prob_drift = A * np.exp(-k * bid_spread_drift)
            elif bid_spread_drift == 0.0:
                ask_prob_drift = A * np.exp(-k * ask_spread_drift)
                bid_prob_drift = 0.0
            else:
                ask_prob_drift = A * np.exp(-k * ask_spread_drift)
                bid_prob_drift = A * np.exp(-k * bid_spread_drift)

            ask_prob_drift = max(0, min(ask_prob_drift, 1))
            bid_prob_drift = max(0, min(bid_prob_drift, 1))

            # Simulate whether a buy or sell order arrives
            ask_action_drift = np.random.binomial(n=1, p=ask_prob_drift)
            bid_action_drift = np.random.binomial(n=1, p=bid_prob_drift)

            # Compute inventory and P&L
            inventory_drift[i] -= ask_action_drift
            pnl_drift[i]       += ask_action_drift * (s + ask_spread_drift)
            inventory_drift[i] += bid_action_drift
            pnl_drift[i]       -= bid_action_drift * (s - bid_spread_drift)

            if i == 0:
                ask_price_drift[t] = s + ask_spread_drift
                bid_price_drift[t] = s - bid_spread_drift
                mid_price_drift[t] = s

        pnl_drift[i] += inventory_drift[i] * s

    # Simulation for the impact model
    print('Impact model:')
    for i in tqdm(range(n_sim)):

        # Stock price for the drift model
        stock_price_impct = np.zeros(T)
        stock_price_impct = np.insert(stock_price_impct, 0, s0)

        for t in range(T):

            # v(t) function
            v_impct = pd.Series(expm(-M_impct * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])

            # Bid and ask spreads
            try:
                ask_spread_impct = 1 / k * np.log(v_impct[f'{inventory_impct[i]}'] / v_impct[f'{inventory_impct[i] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
            except:
                ask_spread_impct = 0.0
            try:
                bid_spread_impct = 1 / k * np.log(v_impct[f'{inventory_impct[i]}'] / v_impct[f'{inventory_impct[i] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
            except:
                bid_spread_impct = 0.0

            # Set bid and ask probabiliy
            if ask_spread_impct == 0.0:
                ask_prob_impct = 0.0                
                bid_prob_impct = A * np.exp(-k * bid_spread_impct)
            elif bid_spread_impct == 0.0:        
                ask_prob_impct = A * np.exp(-k * ask_spread_impct)                
                bid_prob_impct = 0.0
            else:                
                ask_prob_impct = A * np.exp(-k * ask_spread_impct)                
                bid_prob_impct = A * np.exp(-k * bid_spread_impct)

            ask_prob_impct = max(0, min(ask_prob_impct, 1))                    
            bid_prob_impct = max(0, min(bid_prob_impct, 1))

            # Simulate whether a buy or sell order arrives
            ask_action_impct = np.random.binomial(n=1, p=ask_prob_impct)
            bid_action_impct = np.random.binomial(n=1, p=bid_prob_impct)

            # Compute inventory and P&L
            inventory_impct[i] -= ask_action_impct
            pnl_impct[i]       += ask_action_impct * (stock_price_impct[t] + ask_spread_impct)
            inventory_impct[i] += bid_action_impct
            pnl_impct[i]       -= bid_action_impct * (stock_price_impct[t] - bid_spread_impct)

            if i == 0:
                ask_price_impct[t] = stock_price_impct[t] + ask_spread_impct
                bid_price_impct[t] = stock_price_impct[t] - bid_spread_impct
                mid_price_impct[t] = stock_price_impct[t]

            # Update the stock price
            if t != T:
                stock_price_impct[t + 1] = stock_price_impct[t] + sigma * (2 * np.random.binomial(n=1, p=0.5) - 1) + xi * ask_action_impct - xi * bid_action_impct

        pnl_impct[i] += inventory_impct[i] * stock_price_impct[-1]

    # Performance and quotes plots
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.hist(pnl_stand, bins=50, alpha=0.25, color='firebrick', label="Standard")
    plt.hist(pnl_drift, bins=50, alpha=0.25, color='blue', label="Drift")
    plt.hist(pnl_impct, bins=50, alpha=0.25, color='green', label="Impact")
    plt.ylabel('P&l')
    plt.legend()
    ax.set_facecolor('#f2f2f2')
    # plt.title("The P&L histogram of the three strategies")
    plt.savefig('Plots/pnl_glt.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    x = np.arange(T + 1)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(x, ask_price_stand, linewidth=1.0, linestyle="-", label="Ask quote")
    plt.plot(x, bid_price_stand, linewidth=1.0, linestyle="-", label="Bid quote")
    plt.plot(x, mid_price_stand, linewidth=1.0, linestyle="-", label="Mid-price")
    plt.legend()
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # plt.title("Mid-price and optimal bid and ask quotes for the standard strategy")
    plt.savefig('Plots/prices_glt_stand.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    x = np.arange(T + 1)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(x, ask_price_drift, linewidth=1.0, linestyle="-", label="Ask quote")
    plt.plot(x, bid_price_drift, linewidth=1.0, linestyle="-", label="Bid quote")
    plt.plot(x, mid_price_drift, linewidth=1.0, linestyle="-", label="Mid-price")
    plt.legend()
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # plt.title("Mid-price and optimal bid and ask quotes for the drift strategy")
    plt.savefig('Plots/prices_glt_drift.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    x = np.arange(T)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(x, ask_price_impct[:-1], linewidth=1.0, linestyle="-", label="Ask quote")
    plt.plot(x, bid_price_impct[:-1], linewidth=1.0, linestyle="-", label="Bid quote")
    plt.plot(x, mid_price_impct[:-1], linewidth=1.0, linestyle="-", label="Mid-price")
    plt.legend()
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # plt.title("Mid-price and optimal bid and ask quotes for the impact strategy")
    plt.savefig('Plots/prices_glt_impct.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    # Performance summary
    print(f"Mean P&L for the standard strategy: {np.array(pnl_stand).mean()}")
    print(f"Standard deviation of P&L for the standard strategy: {np.sqrt(np.array(pnl_stand).var())}")
    print(f"Mean inventory for the standard strategy: {np.array(inventory_stand).mean()}")
    print(f"Standard deviation of the inventory for the standard strategy: {np.sqrt(np.array(inventory_stand).var())}")
    print('-' * 100)
    print(f"Mean P&L for the drift strategy: {np.array(pnl_drift).mean()}")
    print(f"Standard deviation of P&L for the drift strategy: {np.sqrt(np.array(pnl_drift).var())}")
    print(f"Mean inventory for the drift strategy: {np.array(inventory_drift).mean()}")
    print(f"Standard deviation of the inventory for the drift strategy: {np.sqrt(np.array(inventory_drift).var())}")
    print('-' * 100)
    print(f"Mean P&L for the impact strategy: {np.array(pnl_impct).mean()}")
    print(f"Standard deviation of P&L for the impact strategy: {np.sqrt(np.array(pnl_impct).var())}")
    print(f"Mean inventory for the impact strategy: {np.array(inventory_impct).mean()}")
    print(f"Standard deviation of the inventory for the impact strategy: {np.sqrt(np.array(inventory_impct).var())}")
    print('-' * 100)
    print('Total time elapsed for the simulation: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start_time))))

if __name__ == '__main__':
    main(sys.argv[1:])