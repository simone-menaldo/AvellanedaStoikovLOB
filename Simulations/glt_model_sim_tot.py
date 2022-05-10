#!/usr/bin/python
import sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm
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
mu    = 0.0
xi    = 0.0
sigma = 0.3
q0    = 0
gamma = 0.01
k     = 0.3
A     = 0.5
Q     = 10
seed  = 123

trace = open('Code/Traces/glt_model_sim_tot.txt', 'w')

def main(argv):

    # Import global variables
    global s0, T, mu, xi, sigma, q0, gamma, k, A, Q, seed

    # Define script parameters
    try:
        opts, args = getopt.getopt(argv, 'hp:T:m:x:s:q:g:k:A:Q:d')
    except getopt.GetoptError:
        print('av_model_sim.py -p -T -m -x -s -q -g -k -A -Q -d')
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
            print('d: seed for the simulation')
            sys.exit()
        elif opt == '-p':
            s0 = float(arg)
        elif opt == '-T':
            T = int(arg)
        elif opt == '-m':
            mu = float(arg)
        elif opt == '-x':
            xi = float(arg)
        elif opt == '-s':
            sigma = float(arg)
        elif opt == '-q':
            q0 = int(arg)
        elif opt == '-g':
            gamma = float(arg)
        elif opt == '-k':
            k = float(arg)
        elif opt == '-A':
            A = float(arg)
        elif opt == '-Q':
            Q = int(arg)
        elif opt == '-d':
            seed = int(arg)

    trace.write(f'Model parameters: T={T}, mu={mu}, xi={xi}, sigma={sigma}, gamma={gamma}, k={k}, A={A}, Q={Q}\n')
    trace.write('-' * 100 + '\n')

    # Set the seed
    np.random.seed(seed)

    # Variables holding inventory and P&L
    inventory = [q0] * T
    pnl       = [0] * T

    # Variables holding the price processes
    ask_price = [0] * (T + 1)
    bid_price = [0] * (T + 1)
    mid_price = [0] * (T + 1)

    # Computation of alpha, eta and beta
    alpha = k / 2 * gamma * sigma**2
    eta   = A * (1 + gamma / k)**(-(1 + k / gamma))
    beta  = k * mu

    trace.write(f'Alpha: {alpha}\n')
    trace.write(f'Eta: {eta}\n')
    trace.write(f'Beta: {beta}\n')
    trace.write('-' * 100 + '\n')

    # Computation of the matrix M
    up_diag  = alpha * (Q - np.arange(0, Q + 1))**2 + beta * (Q - np.arange(0, Q + 1))
    low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2 - beta * (Q - np.arange(0, Q)[::-1])
    M        = np.diag(np.concatenate((up_diag, low_diag)))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if j == (i + 1) or j == (i - 1):
                M[i, j] = -eta * np.exp(- k / 2 * xi)

    trace.write('Matrix M:\n')
    trace.write(f'{pd.DataFrame(M)}\n')

    # Stock price initialization
    stock_price = np.zeros(T)
    stock_price = np.insert(stock_price, 0, s0)

    for t in tqdm(range(1, T)):

        trace.write(f'Iteration {t}\n')
        trace.write('-' * 100 + '\n')

        # v(t) function
        v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])

        trace.write(f'v_q(t): {v}\n')
        trace.write('-' * 100 + '\n')

        # Bid and ask spreads
        if inventory[t - 1] > -Q:
            ask_spread = 1 / k * np.log(v[f'{inventory[t]}'] / v[f'{inventory[t] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
        else:
            ask_spread = 0.0

        if inventory[t - 1] < Q:
            bid_spread = 1 / k * np.log(v[f'{inventory[t]}'] / v[f'{inventory[t] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
        else:
            bid_spread = 0.0

        trace.write(f'Ask spread: {ask_spread}\n')
        trace.write(f'Bid spread: {bid_spread}\n')
        trace.write('-' * 100 + '\n')

        # Set bid and ask probabiliy
        if ask_spread == 0.0:
            ask_prob = 0.0                
            bid_prob = A * np.exp(-k * bid_spread)
        elif bid_spread == 0.0:        
            ask_prob = A * np.exp(-k * ask_spread)                
            bid_prob = 0.0
        else:                
            ask_prob = A * np.exp(-k * ask_spread)                
            bid_prob = A * np.exp(-k * bid_spread)

        ask_prob = max(0, min(ask_prob, 1))                    
        bid_prob = max(0, min(bid_prob, 1))

        trace.write(f'Ask prob: {ask_prob}\n')
        trace.write(f'Bid prob: {bid_prob}\n')
        trace.write('-' * 100 + '\n')

        # Simulate whether a buy or sell order arrives
        ask_action = np.random.binomial(n=1, p=ask_prob)
        bid_action = np.random.binomial(n=1, p=bid_prob)

        trace.write(f'Ask action: {ask_action}\n')
        trace.write(f'Bid action: {bid_action}\n')
        trace.write('-' * 100 + '\n')

        # Compute inventory and P&L
        if t != 0:
            inventory[t] = inventory[t - 1] - ask_action + bid_action
            pnl[t]       = pnl[t - 1] + ask_action * (stock_price[t] + ask_spread) - bid_action * (stock_price[t] - bid_spread)

        trace.write(f'Inventory: {inventory[t]}\n')
        trace.write(f'P&L: {pnl[t]}\n')
        trace.write('-' * 100 + '\n')
        
        ask_price[t] = stock_price[t] + ask_spread
        bid_price[t] = stock_price[t] - bid_spread
        mid_price[t] = stock_price[t]

        trace.write(f'Ask price: {ask_price[t]}\n')
        trace.write(f'Bid price: {bid_price[t]}\n')
        trace.write(f'Mid-price: {mid_price[t]}\n')
        trace.write('-' * 100 + '\n')

        # Update the stock price
        if t != T:
            stock_price[t + 1] = stock_price[t] + mu + sigma * (2 * np.random.binomial(n=1, p=0.5) - 1) + xi * ask_action - xi * bid_action

        trace.write(f'Stock price: {stock_price[t]}\n')
        trace.write('-' * 100 + '\n')

    pnl[-1] += inventory[-1] * stock_price[-1]

    trace.write(f'Final P&L: {pnl[t]}')
    trace.write('-' * 100 + '\n')

    # Plot of the optimal quotes and mid-price
    x = np.arange(T)
    plt.plot(x, ask_price[:-1], linewidth=1.0, linestyle="-", label="Ask quote")
    plt.plot(x, bid_price[:-1], linewidth=1.0, linestyle="-", label="Bid quote")
    plt.plot(x, mid_price[:-1], linewidth=1.0, linestyle="-", label="Mid-price")
    plt.xlabel('Time (sec)')
    plt.ylabel('Price')
    plt.legend()
    # caption = f'Mid-price and optimal bid and ask quotes:\n\u03BC={mu}, \u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # plt.text(np.mean(x), min(bid_price) - 5, caption, ha='center', alpha=0.75)
    plt.savefig('Plots/prices_glt_tot.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    # Plot of the inventory dynamics
    x = np.arange(T)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(x, inventory, color='blue', label='inventory')
    plt.axhline(Q, linewidth=1.0, label='Upper bound', color='firebrick')
    plt.axhline(-Q, linewidth=1.0, label='Lower bound', color='firebrick')
    plt.yticks(np.arange(-Q, Q + 1, 2))
    plt.xlabel('Time (sec)')
    plt.ylabel('Inventory')
    plt.legend()
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # caption = f'Dynamics of the inventory:\n\u03BC={mu}, \u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # plt.text(np.mean(x), -Q - 5, caption, ha='center', alpha=0.75)
    plt.savefig('Plots/inventory_glt_tot.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    # Plot of the P&L dynamics
    x = np.arange(T)
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(x, pnl, color='blue')
    plt.xlabel('Time (sec)')
    plt.ylabel('P&L')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # caption = f'Dynamics of the P&L:\n\u03BC={mu}, \u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # plt.text(np.mean(x), min(pnl) - 10, caption, ha='center', alpha=0.75)
    plt.savefig('Plots/pnl_glt_tot.png', bbox_inches='tight', dpi=100, format='png')
    plt.close()

    trace.close()


if __name__ == '__main__':
    main(sys.argv[1:])