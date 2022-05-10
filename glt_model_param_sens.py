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
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Parameters initialization
T     = 600
sigma = 0.3
gamma = 0.01
k     = 0.3
A     = 0.5
mu    = 0.01
xi    = 0.1
Q     = 30

# python glt_model_param_sens.py -p sigma -l 0.1 -u 0.5 -i 0.01
# python glt_model_param_sens.py -p gamma -l 0.005 -u 0.05 -i 0.001
# python glt_model_param_sens.py -p k -l 0.2 -u 0.8 -i 0.01
# python glt_model_param_sens.py -p A -l 0.1 -u 0.7 -i 0.01
# python glt_model_param_sens.py -p mu -l 0.005 -u 0.05 -i 0.001
# python glt_model_param_sens.py -p xi -l 0.05 -u 0.9 -i 0.01

trace = open('Code/Traces/glt_model_param_sens.txt', 'w')

def main(argv):

    # Import global variables
    global T, sigma, gamma, k, A, Q

    try:
        opts, args = getopt.getopt(argv, 'hp:l:u:i:')
    except getopt.GetoptError:
        print('glt_model_param_sens.py -p [sigma, gamma, k, A, mu] -l -u -i')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Parameters for the model')
            print('p: parameter that is allowed to change')
            print('l: upper bound of the range')
            print('u: lower bound of the range')
            print('i: increment of the range')
            sys.exit()
        elif opt == '-p':
            param = str(arg)
        elif opt == '-u':
            upper_bound = float(arg)
        elif opt == '-l':
            lower_bound = float(arg)
        elif opt == '-i':
            increment = float(arg)

    trace.write(f'The floating parameter is {param} in the range [{upper_bound}, {lower_bound}]\n')
    trace.write('-' * 100 + '\n')

    # Variables initialization

    spread_range   = np.arange(lower_bound, upper_bound, increment)

    inventory      = np.arange(-Q, Q + 1)
    bid_spread     = np.zeros(shape=(len(spread_range), 2 * Q + 1))
    ask_spread     = np.zeros(shape=(len(spread_range), 2 * Q + 1))
    bid_ask_spread = np.zeros(shape=(len(spread_range), 2 * Q + 1))

    # Change in sigma
    if param == 'sigma':

        for t, sigma in tqdm(enumerate(spread_range)):
            
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
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('\u03C3')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with \u03C3 and inventory:\nA={A}, k={k}, \u03B3={gamma}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 210)
        plt.savefig('Plots/bid_ask_spread_sigma.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()

    # Change in gamma
    if param == 'gamma':

        for t, gamma in tqdm(enumerate(spread_range)):
            
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
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('\u03B3')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with \u03B3 and inventory:\n\u03C3={sigma}, A={A}, k={k}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 30)
        plt.savefig('Plots/bid_ask_spread_gamma.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()

    # Change in A
    if param == 'A':

        for t, A in tqdm(enumerate(spread_range)):
            
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
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('A')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with A and inventory:\n\u03C3={sigma}, k={k}, \u03B3={gamma}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 30)
        plt.savefig('Plots/bid_ask_spread_A.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()

    # Change in k
    if param == 'k':

        for t, k in tqdm(enumerate(spread_range)):
            
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
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('k')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with k and inventory:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 30)
        plt.savefig('Plots/bid_ask_spread_k.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()

    # Change in mu
    if param == 'mu':

        for t, mu in tqdm(enumerate(spread_range)):
            
            # Computation of alpha, beta, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            beta  = k * mu
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            trace.write(f'Alpha: {alpha}\n')
            trace.write(f'Beta: {beta}\n')
            trace.write(f'Eta: {eta}\n')
            trace.write('-' * 100 + '\n')

            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i - 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i]     = 1 / k * (np.log(v[i]) - np.log(v[i + 1])) + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('\u03BC')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with \u03BC and inventory:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 30)
        plt.savefig('Plots/bid_ask_spread_mu.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()

    # Change in xi
    if param == 'xi':

        for t, xi in tqdm(enumerate(spread_range)):
            
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
                        M[i, j] = -eta

            trace.write('Matrix M:\n')
            trace.write(f'{pd.DataFrame(M)}\n')

            # v(t) function
            v = expm(-M * T) @ np.ones(len(range(-Q, Q+1))).T

            trace.write(f'v_q(t): {v}\n')
            trace.write('-' * 100 + '\n')

            # Bid, ask and bid-ask spreads
            for i in range(1, 2 * Q + 1):
                ask_spread[t, i] = 1 / k * np.log(v[i]) - np.log(v[i - 1]) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
            for i in range(2 * Q):
                bid_spread[t, i] = 1 / k * np.log(v[i]) - np.log(v[i + 1]) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
            for i in range(1, 2 * Q):
                bid_ask_spread[t, i] = - 1 / k * (np.log(v[i + 1]) + np.log(v[i - 1]) - np.log(v[i]**2)) + xi + 2 / gamma * np.log(1 + gamma / k)

            trace.write(f'Ask spread:     {ask_spread[t, :]}\n')
            trace.write(f'Bid spread:     {bid_spread[t, :]}\n')
            trace.write(f'Bid-ask spread: {bid_ask_spread[t, :]}\n')
            trace.write('-' * 100 + '\n')

        s, q = np.meshgrid(spread_range, inventory[6:-6])

        # Plot of the bid-ask spread
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf    = ax.plot_surface(s, q, bid_ask_spread[:, 6:-6].T, color='firebrick', alpha=0.75)
        ax.set_xlabel('\u03BE')
        ax.set_ylabel('Inventory')
        ax.set_zlabel('Bid-ask spread (Tick)')
        # caption = f'Asymptotic behaviour of the optimal bid-ask spread with \u03BE and inventory:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}'
        # fig.text(0.5, 0.05, caption, ha='center')
        ax.view_init(30, 210)
        plt.savefig('Plots/bid_ask_spread_xi.png', bbox_inches='tight', dpi=100, format='png')

        trace.close()


if __name__ == '__main__':
    main(sys.argv[1:])