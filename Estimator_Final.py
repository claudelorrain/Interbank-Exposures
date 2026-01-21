import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import mvn
from itertools import product 
from scipy.optimize import approx_fprime
from scipy.optimize import differential_evolution
import networkx as nx
import matplotlib.pyplot as plt
import cma
import os
import sys
from datetime import datetime


# --------------------- Fetch Cleaned Data ---------------------------
print("Importing.....")

os.chdir("path-to-file")

raw_means = pd.read_csv("Mean_Prices.csv")
raw_vcov = pd.read_csv("Variance_Covariance_Log_Prices.csv")
raw_pd = pd.read_csv("CDS_PD.csv")
raw_assets = pd.read_csv("Permco_Cusip_Link.csv")

ordered_cusips = raw_assets['cusip'].tolist()
n_banks = len(ordered_cusips)

cusip_to_index = {cusip: cusipx for cusipx, cusip in enumerate(ordered_cusips)}
lognormal_assets = raw_assets['totast'].to_numpy() * 1000  # Already ordered, this order is taken as default ordering

# Initialize covariance matrix
normal_vcov = np.zeros((n_banks, n_banks))

# Fill in the covariance matrix
for d, row in raw_vcov.iterrows():
    id1 = row['cusip1']
    id2 = row['cusip2']
    i = cusip_to_index[id1] # Ensure same ordering as assets
    j = cusip_to_index[id2]
    if i is not None and j is not None:
        normal_vcov[i, j] = row['covariance']

prob_def = raw_pd
prob_def['order'] = raw_pd['cusip'].map(cusip_to_index)
prob_def = prob_def.sort_values('order').drop(columns=['order', 'cusip', 'ticker']).reset_index(drop=True)
prob_def = prob_def['mean_prob'].tolist()

lognormal_means = raw_means
lognormal_means['order'] = raw_means['cusip'].map(cusip_to_index)
lognormal_mean = lognormal_means.sort_values('order').drop(columns=['order', 'cusip']).reset_index(drop=True)
lognormal_mean = lognormal_mean['mean'].tolist()

# Smaller selection of banks
subsample = 10
prob_def = prob_def[:subsample]
normal_vcov = 1* normal_vcov[:subsample, :subsample]
lognormal_mean = lognormal_mean[:subsample]
ordered_cusips = ordered_cusips[:subsample]
lognormal_assets = lognormal_assets[:subsample]*1000  # Convert to USD

print("Ordered CUSIPs:\n", ordered_cusips)
print("Prob Def:\n", prob_def)
print("Total Probability of Default:", np.sum(prob_def))
print("Lognormal Mean\n:", lognormal_mean)
print("Normal Variance Covariance Matrix:\n", normal_vcov)


"""# --------------------------------- Test Data ---------------------------------

prob_def = [0.003, 0.005, 0.0002, 0.0008, 0.004]
print("Total Probability of Default:", np.sum(prob_def))
normal_vcov = np.identity(5)*0.07
normal_vcov = (normal_vcov + normal_vcov.T)/2
lognormal_mean = [3*10**12, 5*10**12, 7*10**12, 8*10**11, 3*10**12]
ordered_cusips = ["A", "B", "C", "D", "E"]"""
# ----------------------------------------- Preliminary -------------------------------------------
print("Data imported.")
# Ensure these are Pr(e1<k) for some strike k
marginals = prob_def 

n_banks = len(prob_def)
delta = 0.4

# Note ln(w) for w<=0 is treated as -inf.


# ------------------------------- Compute Joint Probabilities -----------------------------

states = list(product([0, 1], repeat=n_banks)) # Output is e.g. [(0,0,0), (0,0,1), (0,1,0),...] 
cases = np.array(states, dtype=int)
cases_sorted = sorted(cases, key=lambda x: sum(x))


def compute_joint_probabilities(L, samples):
    # Samples must be normal
    # Go through the set of cases one by one. For each case, compute the threshold according to the formula
    joint_probabilities = np.zeros(2**n_banks)
    delta_matrix = (1 - delta) * L.T # Each entry i,j is what bank i expects to lose upon bank j's default 
    w = L.sum(axis=1) - L.sum(axis=0) # Base cases

    for k, case in enumerate(cases):
        total_loss = np.sum(delta_matrix * case[np.newaxis, :], axis=1) # Multiply to find the total losses to each bank conditional on this particular default set

        thresholds = w + total_loss # Thresholds for each bank 
        thresholds_log = np.full_like(thresholds, -np.inf, dtype=float)
        positive_mask = thresholds > 0
        thresholds_log[positive_mask] = np.log(thresholds[positive_mask])
        thresholds = thresholds_log
        # print("Thresholds\n", thresholds)
        logw = np.full_like(w, -np.inf, dtype=float)
        positive_mask = w > 0
        logw[positive_mask] = np.log(w[positive_mask])
        logw = logw

        greater = samples > thresholds
        less_equal = samples <= thresholds

        mask = np.where(case, less_equal, greater) # 1 means apply less_equal, 0 means apply greater
        joint_mask = np.all(mask, axis=1) # All n must jointly apply

        joint = np.mean(joint_mask)

        # Now subtract the joint probability of all banks surviving under this case, since that has been counted accidentally

        mask2 = np.zeros((samples.shape[0], n_banks), dtype=bool)
        if np.sum(case) > 0:
            for i in range(n_banks):
                    if case[i] == 0:
                        mask2[:,i] = greater[:, i]
                    else:
                        mask2[:,i] = (samples[:, i] > logw[i]) & (samples[:, i] <= thresholds[i])
        
        joint_mask_sub = np.all(mask2, axis=1) # All n must jointly apply
        joint_mask_sub = np.mean(joint_mask_sub)

        joint_probabilities[k] = joint - joint_mask_sub

    return joint_probabilities

# ------------------------------ Error Function to Minimise ---------------------------------
def objective(L_vec):

    L = L_vec.reshape(n_banks,n_banks)

    # Converting means
    # mean_est = np.where(lognormal_assets - L.sum(axis=0) > 0, lognormal_assets - L.sum(axis=0), lognormal_assets) # Total assets - estimated claims on other bank assets is an approximation for the mean of self-owned asset value
    # mu = np.log(lognormal_mean) - 0.5*np.diag(normal_vcov)
    # print("Normal means\n", mu)
    # print("Lognormal mean\n", lognormal_mean)
    # mean_est = lognormal_mean + np.where(L.sum(axis=0) - L.sum(axis=1) > 0, L.sum(axis=0) - L.sum(axis=1), 0)
    #  
    mean_est = np.where(lognormal_mean + L.sum(axis=0) - L.sum(axis=1) >0, lognormal_mean + L.sum(axis=0) - L.sum(axis=1), 1e7)
    mu = np.log(mean_est) - 0.5*np.diag(normal_vcov)
    # print("Mean\n", mu)
    sigma = normal_vcov
    # print("Variances\n", np.diag(normal_vcov))
    # ------------------------- MONTE CARLO SAMPLES ---------------------------------
    N_SAMPLES = 10000
    RANDOM_SEED = 42

    normal_dist = multivariate_normal(mean=mu, cov=sigma)
    normal_samples = normal_dist.rvs(size=N_SAMPLES, random_state=RANDOM_SEED)


    # Given a particular vector of strikes, we want the implied and actual probabilities:
    joints = compute_joint_probabilities(L, normal_samples) # Matched to each of the cases
    # print("Joints\n", joints)
    predicted_marginals = np.dot(joints, cases_sorted)
    # print("Predicted Marginals\n", predicted_marginals)
    # print("Actual Marginals\n", prob_def)
    midpoint = (predicted_marginals + prob_def) / 2
    # total_error = np.sum(jsd_binary(predicted_marginals, prob_def)) # Total shannon divergence
    # total_error = np.sum((predicted_marginals - prob_def)**2) # Total squared error
    total_error = np.sum(np.abs(predicted_marginals - prob_def)) # Total absolute error

    return total_error 

def jsd_binary(p, q, eps=1e-12):
    # Clip to avoid log(0) or division by 0
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)

    m = 0.5 * (p + q)

    kl_pm = p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m))
    kl_qm = q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))

    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


# bounds = [(0, 0) if i == j else (0, 100000000000) for i in range(n_banks) for j in range(n_banks)]
print("Start optimisation")
#----------------------------------- OPTIMISATION -------------------------------------------
# OPTIMIZER 3: Covariance Matrix Adaptation 
rng = np.random.default_rng(seed=10)
initial_guess = rng.integers(low=1e8, high=5e9, size=(n_banks,n_banks))
np.fill_diagonal(initial_guess, 0) # Random initial guess
flat = initial_guess.flatten("C")

if __name__ == "__main__":
    mask = ~np.eye(n_banks, dtype=bool)
    initial_guess_flat = flat[mask.flatten()]
    epsilon = 1e-8
    lower_bounds = [0 if i == j else 1e7 for i in range(n_banks) for j in range(n_banks)]
    upper_bounds = [epsilon if i == j else 1e11 for i in range(n_banks) for j in range(n_banks)]


    sig = 7e10
    # Increase verbosity for CMA-ES output
    es = cma.CMAEvolutionStrategy(flat.tolist(), sig, {
        'bounds': [lower_bounds, upper_bounds],
        'seed': 42,
        'maxiter': 100,
        'verb_disp': 2,      
        'verbose': 2,        
        'tolfun': 1e-15,
        'tolx': 1e-15,
        'verb_log': 0,  
        'maxfevals': 1e5,
        'popsize': 60,
        'tolflatfitness': 200 
    })

    es.optimize(lambda x: objective(np.array(x)))

    # Recover full L matrix
    print("Stop reason:", es.stop())
    L_est = np.array(es.result.xbest)
    L_est = np.reshape(L_est, (n_banks,n_banks))
    print("Estimated L matrix:\n", L_est)
    print("Initial guess:\n", initial_guess)
    dev = np.sum(L_est - initial_guess)

    print("Optimisation success!")
    print("\n--- Estimated L Matrix (CMA-ES) ---")
    df = pd.DataFrame(L_est, index=[f'From {i+1}' for i in range(n_banks)], columns=[f'To {j+1}' for j in range(n_banks)])
    print(df.round(4))

    final_sse = objective(L_est.flatten())
    print(f"\nFinal Objective Function Value: {final_sse:.6f}")
    print(f"Deviation from initial guess: {dev:.6f}")

    l_est_df = pd.DataFrame(L_est, index=ordered_cusips, columns=ordered_cusips)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    links_filename = f'output_links_{timestamp}.csv'
    l_est_df.to_csv(links_filename)
    print("Saved output.")

"""
# ---------------------- Draw Networks -------------------------------
def plot_network(ax, matrix, title):

    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)

    # --- Edge Thickness Scaling ---
    all_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    edge_widths = []

    if all_weights:
        min_w, max_w = min(all_weights), max(all_weights)
        min_width, max_width = 1.0, 10.0 # Set a clear min/max width

        if max_w == min_w:
            edge_widths = [min_width] * len(all_weights)
        else:
            edge_widths = [
                min_width + ((w - min_w) / (max_w - min_w)) * (max_width - min_width)
                for w in all_weights
            ]

    # --- Node and Edge Drawing ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color='red',
        node_size=1500, # Increased size to better match example
        edgecolors='red'
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color='white',
        font_weight='normal',
        font_size=14
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='black',
        width=edge_widths,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=25,
        connectionstyle='arc3,rad=0.2' # Increased radius for more curve
    )

    # --- Edge Label Drawing (with fixes) ---

    # 1. Format labels to round floats for readability
    formatted_edge_labels = {
        (u, v): f"{d['weight']:.2f}" if isinstance(d['weight'], float) else f"{d['weight']}"
        for u, v, d in G.edges(data=True)
    }

    # 2. Draw labels with offsets to prevent overlap
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=formatted_edge_labels,
        # KEY FIX: Move label position away from the center (0.5)
        label_pos=0.3,
        font_size=10,
        font_color='black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
    )

    ax.set_title(title, fontsize=20, pad=20)
    ax.axis('off')



fig, axes = plt.subplots(1, 2, figsize=(16, 8))

plot_network(axes[0], initial_guess, "Network from initial guess")
plot_network(axes[1], L_est, "Network from final estimate")

plt.tight_layout()

plt.show()"""
