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
plt.rcParams['font.size'] = 23
plt.rcParams['axes.labelpad'] = 18
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Parameters initialization
T     = 600
xi    = 0.5
sigma = 0.3
gamma = 0.01
k     = 0.3
A     = 0.5
Q     = 30

trace = open('Code/Traces/glt_model_spread_imp.txt', 'w')

def main(argv):

    # Import global variables
    global T, xi, sigma, gamma, k, A, Q

    # Define script parameters
    try:
        opts, args = getopt.getopt(argv, 'hT:x:s:g:k:A:Q:')
    except getopt.GetoptError:
        print('glt_model_spread_imp.py -T -x -s -g -k -A -Q')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Parameters for the model')
            print('T: final time of the simulation')
            print('x: market impact parameter')
            print('s: intraday volaility of the stock price')
            print('g: risk adversion')
            print('k: price sensitivity of market participants')
            print('A: order rate arrival multiplier')
            print('Q: maximum inventory')
            sys.exit()
        elif opt == '-T':
            T = int(arg)
        elif opt == '-x':
            xi = float(arg)
        elif opt == '-s':
            sigma = float(arg)
        elif opt == '-g':
            gamma = float(arg)
        elif opt == '-k':
            k = float(arg)
        elif opt == '-A':
            A = float(arg)
        elif opt == '-Q':
            Q = int(arg)

    trace.write(f'Model parameters: T={T}, xi={xi}, sigma={sigma}, gamma={gamma}, k={k}, A={A}, Q={Q}\n')
    trace.write('-' * 100 + '\n')

    # Variables initialization

    inventory      = np.arange(-Q, Q + 1)
    bid_spread     = np.zeros(shape=(T, 2 * Q + 1))
    ask_spread     = np.zeros(shape=(T, 2 * Q + 1))
    bid_ask_spread = np.zeros(shape=(T, 2 * Q + 1))

    # Computation of alpha, eta and matrix M
    alpha = k / 2 * gamma * sigma**2
    eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

    trace.write(f'Alpha: {alpha}\n')
    trace.write(f'Eta: {eta}\n')
    trace.write('-' * 100 + '\n')

    up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
    low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
    M        = np.diag(np.concatenate((up_diag, low_diag)))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if j == (i + 1) or j == (i - 1):
                M[i, j] = -eta * np.exp(- k / 2 * xi)

    trace.write('Matrix M:\n')
    trace.write(f'{pd.DataFrame(M)}\n')

    for t in tqdm(range(T)):

        trace.write(f'Iteration {t}\n')
        trace.write('-' * 100 + '\n')

        # v(t) function
        v = expm(-M * (T - t)) @ np.ones(len(range(-Q, Q+1))).T

        trace.write(f'v_q(t): {v}\n')
        trace.write('-' * 100 + '\n')

        # Bid, ask and bid-ask spreads
        for i in range(1, 2 * Q + 1):
            ask_spread[t, i] = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
        for i in range(2 * Q):
            bid_spread[t, i] = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
        for i in range(1, 2 * Q):
            bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + xi + 2 / gamma * np.log(1 + gamma / k)

        trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
        trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
        trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
        trace.write('-' * 100 + '\n')

    # Asymptotic bid and ask spreads
    bid_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + xi / 2 + (2 * inventory + 1) / 2 * np.exp(k / 4 * xi) * np.sqrt((sigma**2 * k) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))
    ask_spread_aysm = 1 / gamma * np.log(1 + gamma / k) + xi / 2 - (2 * inventory - 1) / 2 * np.exp(k / 4 * xi) * np.sqrt((sigma**2 * k) / (2 * k * A) * (1 + gamma / k)**(1 + k / gamma))

    trace.write(f'Asymptotic bid spread: {bid_spread_aysm}\n')
    trace.write(f'Asymptotic ask spread: {ask_spread_aysm}\n')
    trace.write('-' * 100 + '\n')

    t = np.arange(T)
    t, q = np.meshgrid(t[::-1], inventory[2:-2])

    # Plot of the bid spread
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf    = ax.plot_surface(t, q, bid_spread[:, 2:-2].T, color='firebrick', alpha=0.75)
    ax.set_xticks(list(ax.get_xticks())[1:-1])
    ax.set_xticklabels([str(i) for i in np.arange(0, T + 100, 100)[::-1]])
    ax.set_xlabel('Time (Sec)')
    ax.set_ylabel('Inventory')
    ax.set_zlabel('Bid spread (Tick)')
    # caption = f'Behaviour of the optimal bid quote with time and inventory:\n\u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # fig.text(0.5, 0.05, caption, ha='center')
    ax.view_init(30, -30)
    plt.savefig('Plots/bid_spread_imp.png', bbox_inches='tight', dpi=100, format='png')

    t = np.arange(T)
    t, q = np.meshgrid(t[::-1], inventory[2:-2])

    # Plot of the ask spread
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf    = ax.plot_surface(t, q, ask_spread[:, 2:-2].T, color='firebrick', alpha=0.75)
    ax.set_xticks(list(ax.get_xticks())[1:-1])
    ax.set_xticklabels([str(i) for i in np.arange(0, T + 100, 100)[::-1]])
    ax.set_xlabel('Time (Sec)')
    ax.set_ylabel('Inventory')
    ax.set_zlabel('Ask spread (Tick)')
    # caption = f'Behaviour of the optimal ask quote with time and inventory:\n\u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # fig.text(0.5, 0.05, caption, ha='center')
    ax.view_init(30, 30)
    plt.savefig('Plots/ask_spread_imp.png', bbox_inches='tight', dpi=100, format='png')

    t = np.arange(T)
    t, q = np.meshgrid(t, inventory[2:-2])

    # Plot of the bid-ask spread
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surf    = ax.plot_surface(t, q, bid_ask_spread[:, 2:-2].T, color='firebrick', alpha=0.75)
    ax.set_xlabel('Time (Sec)')
    ax.set_ylabel('Inventory')
    ax.set_zlabel('Bid-ask spread (Tick)')
    # caption = f'Behaviour of the optimal bid-ask spread with time and inventory:\n\u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # fig.text(0.5, 0.05, caption, ha='center')
    ax.view_init(30, 210)
    plt.savefig('Plots/bid_ask_spread_imp.png', bbox_inches='tight', dpi=100, format='png')

    # Plot of the asymptotic bid spread
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(inventory[:-1], bid_spread[0, :-1], linestyle='-', linewidth=3.0, color='firebrick')
    plt.plot(inventory, bid_spread_aysm, 'k--')
    plt.xlabel('Inventory')
    plt.ylabel('Bid spread')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # caption = f'Asymptotic behaviour of the optimal bid quote:\n\u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # plt.figtext(0.5, 0, caption, ha='center')
    plt.savefig('Plots/bid_spread_aysm_imp.png', bbox_inches='tight', dpi=100, format='png')

    # Plot of the asymptotic ask spread
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    plt.plot(inventory[1:], ask_spread[0, 1:], linestyle='-', linewidth=3.0, color='firebrick')
    plt.plot(inventory, ask_spread_aysm, 'k--')
    plt.xlabel('Inventory')
    plt.ylabel('Ask spread')
    plt.grid()
    ax.set_facecolor('#f2f2f2')
    # caption = f'Asymptotic behaviour of the optimal ask quote:\n\u03BE={xi}, \u03C3={sigma}, A={A}, k={k}, \u03B3={gamma}, T={T}'
    # plt.figtext(0.5, 0, caption, ha='center')
    plt.savefig('Plots/ask_spread_aysm_imp.png', bbox_inches='tight', dpi=100, format='png')

    trace.close()


if __name__ == '__main__':
    main(sys.argv[1:])