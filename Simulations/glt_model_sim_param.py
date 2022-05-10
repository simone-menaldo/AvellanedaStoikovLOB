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
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelpad'] = 20
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'dimgray'

# Parameters initialization
T     = 600
s0    = 100
sigma = 0.3
gamma = 0.01
k     = 0.3
A     = 0.5
mu    = 0.0
xi    = 0.0
q0    = 0
Q     = 30
seed  = 123

# python glt_model_sim_param.py -p sigma -l 0.1 -u 0.5 -i 0.01
# python glt_model_sim_param.py -p gamma -l 0.005 -u 0.05 -i 0.001
# python glt_model_sim_param.py -p k -l 0.2 -u 0.8 -i 0.01
# python glt_model_sim_param.py -p A -l 0.1 -u 0.7 -i 0.01
# python glt_model_sim_param.py -p mu -l 0.005 -u 0.05 -i 0.001
# python glt_model_sim_param.py -p xi -l 0.05 -u 0.9 -i 0.01

def main(argv):

    # Import global variables
    global T, s0, sigma, gamma, k, A, mu, xi, q0, Q, seed

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

    # Set the seed
    np.random.seed(seed)

    # Variables initialization
    param_range   = np.arange(lower_bound, upper_bound, increment)

    inventory = [q0] * len(param_range)
    pnl       = [0] * len(param_range)

    # Change in sigma
    if param == 'sigma':

        for p, sigma in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('\u03C3')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating \u03C3:\nA={A}, k={k}, \u03B3={gamma}, T={T}, \u03BC={mu}, \u03BE={xi}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')

    # Change in gamma
    if param == 'gamma':

        for p, gamma in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('\u03B3')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating \u03B3:\n\u03C3={sigma}, A={A}, k={k}, T={T}, \u03BC={mu}, \u03BE={xi}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')

    # Change in A
    if param == 'A':

        for p, A in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('A')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating A:\n\u03C3={sigma}, k={k}, \u03B3={gamma}, T={T}, \u03BC={mu}, \u03BE={xi}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')

    # Change in k
    if param == 'k':

        for p, k in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('k')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating k:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}, \u03BC={mu}, \u03BE={xi}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')

    # Change in mu
    if param == 'mu':

        for p, mu in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('\u03BC')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating \u03BC:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}, \u03BE={xi}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')

    # Change in xi
    if param == 'xi':

        for p, xi in tqdm(enumerate(param_range)):
            
            # Computation of alpha, eta and matrix M
            alpha = k / 2 * gamma * sigma**2
            eta   = A * (1 + gamma / k)**(-(1 + k / gamma))

            # Computation of matrix M
            up_diag  = alpha * (Q - np.arange(0, Q + 1))**2
            low_diag = alpha * (Q - np.arange(0, Q)[::-1])**2
            M        = np.diag(np.concatenate((up_diag, low_diag)))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if j == (i + 1) or j == (i - 1):
                        M[i, j] = -eta

            # Stock price initialization
            stock_price = np.zeros(T)
            stock_price = np.insert(stock_price, 0, s0)

            for t in range(1, T):

                # v(t) function
                v = pd.Series(expm(-M * (T - t)) @ np.ones(2 * Q + 1).T, index=['{}'.format(j) for j in np.arange(-Q, Q+1)])
    
                # Bid and ask spreads
                if inventory[p] > -Q:
                    ask_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] - 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    ask_spread = 0.0
    
                if inventory[p] < Q:
                    bid_spread = 1 / k * np.log(v[f'{inventory[p]}'] / v[f'{inventory[p] + 1}']) + xi / 2 + 1 / gamma * np.log(1 + gamma / k)
                else:
                    bid_spread = 0.0
    
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
    
                # Simulate whether a buy or sell order arrives
                ask_action = np.random.binomial(n=1, p=ask_prob)
                bid_action = np.random.binomial(n=1, p=bid_prob)

                # Update inventory and P&L
                inventory[p] -= ask_action
                pnl[p]       += ask_action * (stock_price[t] + ask_spread)
                inventory[p] += bid_action
                pnl[p]       -= bid_action * (stock_price[t] - bid_spread)

            pnl[p] += inventory[p] * stock_price[-1]

        # Histogram of the P&L with the floating parameter
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)
        plt.bar(param_range, pnl, width=increment/2, color='firebrick')
        plt.xlabel('\u03BE')
        plt.ylabel('P&L')
        ax.set_facecolor('#f2f2f2')
        # caption = f'Change in P&L with floating \u03BE:\n\u03C3={sigma}, A={A}, \u03B3={gamma}, T={T}, \u03BC={mu}'
        # plt.figtext(0.5, -0.02, caption, ha='center', alpha=0.75)
        plt.savefig(f'Plots/pnl_glt_{param}.png', bbox_inches='tight', dpi=100, format='png')


if __name__ == '__main__':
    main(sys.argv[1:])