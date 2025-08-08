import DeepLINK as dl
from PCp1_numFactors import PCp1 as PCp1
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from pairwise_connected_layer import PairwiseConnected
from itertools import combinations
from keras.callbacks import EarlyStopping
import tensorflow as tf
import random
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

count =  pd.read_csv("/Users/edwardpeng/Desktop/RA/zinckAA_sim/count.csv")
count = count.values  # convert to array

### data generating process ###
def generate_data_AA(p, pos, u, seed):
    np.random.seed(seed)
    # Order columns by decreasing abundance
    col_sums = np.sum(count, axis=0)
    sorted_indices = np.argsort(-col_sums)  # Descending order
    dcount = count[:, sorted_indices]

    # Filter features with >20% non-zero samples
    norm_count = count / np.sum(count, axis=1, keepdims=True)
    col_means = np.mean(norm_count > 0, axis=0)
    valid_indices = np.where(col_means > 0.2)[0]
    sorted_valid_indices = valid_indices[np.argsort(-col_means[valid_indices])]

    # Select top p features (p could be 200, 300, 400)
    dcount = count[:, sorted_valid_indices][:, :p]

    # Randomly sample 500 observations
    n_total = dcount.shape[0]
    sel_index = np.sort(np.random.choice(n_total, size=500, replace=False))
    dcount = dcount[sel_index, :]

    # Add pseudocount and calculate proportions
    original_OTU = dcount + 0.5
    seq_depths = np.sum(original_OTU, axis=1)
    Pi = original_OTU / seq_depths[:, np.newaxis]
    n = Pi.shape[0]

    # Generate binary responses (50% cases, 50% controls)
    Y = np.array([0] * 250 + [1] * 250)
    np.random.shuffle(Y)  # Shuffle to randomize assignment

    # Randomly select 30 biomarkers from top 200 abundant features
    biomarker_idx = np.random.choice(200, size=30, replace=False)

    # Assign effect directions
    n_positive = int(np.round(30 * pos / 100))
    n_negative = 30 - n_positive
    signs = np.array([1] * n_positive + [-1] * n_negative)
    np.random.shuffle(signs)

    # Modify proportions for selected biomarkers
    Delta = 3  # Could be adjusted to have enough power
    fold_changes = np.random.uniform(0, Delta, size=30)
    Pi_new = Pi.copy()

    for j in range(30):
        col_j = biomarker_idx[j]
        if signs[j] == 1:
            # Positive effect: increase in cases
            Pi_new[Y == 1, col_j] = Pi[Y == 1, col_j] * (1 + fold_changes[j])
        else:
            # Negative effect: increase in controls
            Pi_new[Y == 0, col_j] = Pi[Y == 0, col_j] * (1 + fold_changes[j])
            
    # Renormalize proportions
    Pi_new = Pi_new / np.sum(Pi_new, axis=1, keepdims=True)
    
    # Draw sequencing depths from template data
    template_seq_depths = np.sum(count, axis=1)
    drawn_depths = np.random.choice(template_seq_depths, size=n, replace=True)
    adjusted_depths = drawn_depths.copy()
    adjusted_depths[Y == 1] = drawn_depths[Y == 1] * (1 + u)  # Apply depth increase to case group

    sim_count = np.zeros((n, p), dtype=int)
    
    for i in range(n):
        # Use multinomial to generate counts for sample i
        sim_count[i, :] = np.random.multinomial(
            n=int(adjusted_depths[i]), 
            pvals=Pi_new[i, :], 
            size=1
        ).flatten()
    
    return {
        'Y': Y,
        'X': sim_count,
        'signal_indices': biomarker_idx
    }

aut_epoch = 100 # number of autoencoder training epochs
aut_loss = 'mean_squared_error' # loss function used in autoencoder training
aut_verb = 0 # verbose level of autoencoder
mlp_epoch = 100 # number of mlp training epochs
mlp_loss = 'binary_crossentropy'
#mlp_loss = 'mean_squared_error' # loss function used in mlp training
dnn_loss = 'binary_crossentropy'
dnn_verb = 0
aut_met = 'relu'
dnn_met = 'elu'
mlp_verb = 0 # verbose level of mlp
l1 = 0.001 # l1 regularization factor in mlp
lr = 0.001 # learning rate for mlp training

def run_deeplink(X, Y, target_fdr):
    try:
        # Standardize X
        X_std = X.copy()
        X_std = X_std - np.mean(X_std, axis=0)
        std_dev = np.std(X_std, axis=0, ddof=1)
        std_dev[std_dev == 0] = 1
        X_std = X_std / std_dev

        # Normalize X1
        norms = np.sqrt(np.sum(X_std ** 2, axis=0))
        norms[norms == 0] = 1
        X1 = X_std / norms
        # Estimate r_hat
        r_hat = PCp1(X1, 15)
        # generate knockoff
        Xnew = dl.knockoff_construct(X_std, r_hat, aut_met, aut_epoch, aut_loss, aut_verb)

        n_features = Xnew.shape[1] // 2
        
        # implement DeepPINK
        es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
        dp = Sequential()
        dp.add(PairwiseConnected(input_shape=(2 * n_features,)))
        dp.add(Dense(n_features, activation='elu', kernel_regularizer=keras.regularizers.l1(l1=0.001)))
        dp.add(Dense(1, activation='relu',
                     kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
        dp.compile(loss=mlp_loss, optimizer=keras.optimizers.Adam(learning_rate=lr))
        dp.fit(Xnew, Y, epochs=mlp_epoch, batch_size=32, verbose=mlp_verb, validation_split=0.1, callbacks=[es])

        weights = dp.get_weights()
        w = weights[1] @ weights[3]
        w = w.reshape(n_features, )
        z = weights[0][:n_features]
        z_tilde = weights[0][n_features:]
        W = (w * z) ** 2 - (w * z_tilde) ** 2
        
        # Feature selection
        selected = dl.knockoff_select(W, target_fdr, ko_plus=False)
        
        return selected
        
    except Exception as e:
        print(f"DeepLINK failed: {e}")
        return np.array([])

def calculate_metrics(selected, signal_indices):
    n_selected = len(selected)
    
    if n_selected == 0:
        empirical_fdr = 0.0
        power = 0.0
    else:
        true_positives = len(set(selected) & set(signal_indices))
        false_positives = n_selected - true_positives
        empirical_fdr = false_positives / n_selected if n_selected > 0 else 0.0
        power = true_positives / len(signal_indices) if len(signal_indices) > 0 else 0.0
    
    return empirical_fdr, power, n_selected

def run_single_simulation(params):
    p, pos, u, sim_id = params
    # Set unique seed for this simulation
    seed = 12345 + sim_id * 1000 + p + pos * 10 + int(u * 100)

    try:
        data = generate_data_AA(p, pos, u, seed)
        X, Y, signal_indices = data['X'], data['Y'], data['signal_indices']
        results = []
        target_fdrs = [0.05, 0.1, 0.15, 0.2]

        for target_fdr in target_fdrs:
            selected = run_deeplink(X, Y, target_fdr)
            empirical_fdr, power, n_selected = calculate_metrics(selected, signal_indices)
            
            results.append({
                'p': p,
                'pos': pos,
                'u': u,
                'sim': sim_id,
                'target_fdr': target_fdr,
                'method': 'DeepLINK',
                'empirical_fdr': empirical_fdr,
                'power': power,
                'n_selected': n_selected
            })

        return results

    except Exception as e:
        print(f"Simulation failed for params {params}: {e}")
        return []

def run_simulation_for_combination(combo_params):
    p, pos, u = combo_params
    print(f"Starting simulations for p={p}, pos={pos}%, u={u}")
    
    # Create parameter list for all 100 simulations
    params_list = [(p, pos, u, sim_id) for sim_id in range(1, 101)]
    
    # Run 100 simulations in parallel using 8 cores
    with mp.Pool(processes=8) as pool:
        all_results_nested = pool.map(run_single_simulation, params_list)
    
    # Flatten the nested results
    all_results = []
    for sim_results in all_results_nested:
        all_results.extend(sim_results)
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results for this combination
    filename = f"simulation_results_p{p}_pos{pos}_u{u}.csv"
    results_df.to_csv(filename, index=False)
    print(f"Completed and saved results for p={p}, pos={pos}%, u={u} to {filename}")
    
    return results_df

def main():
    p_values = [200, 300, 400]
    pos_values = [40, 80, 100]
    u_values = [0, 0.5, 1.0]
    
    combinations = [(p, pos, u) for p in p_values for pos in pos_values for u in u_values]
    print(f"Total parameter combinations: {len(combinations)}")
    print("Each combination will run 100 simulations in parallel using 8 cores")
    
    # Run simulations sequentially across combinations 
    # (each combination uses 8 cores internally)
    all_combination_results = []
    for i, combo in enumerate(combinations, 1):
        print(f"\n=== Processing combination {i}/{len(combinations)} ===")
        result_df = run_simulation_for_combination(combo)
        all_combination_results.append(result_df)
    
    # Combine all results
    final_results = pd.concat(all_combination_results, ignore_index=True)
    
    # Save combined results
    final_results.to_csv("complete_simulation_results.csv", index=False)
    print(f"\n=== All simulations completed! ===")
    print(f"Total rows in final results: {len(final_results)}")
    
    return final_results

if __name__ == "__main__":
    # Set up for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # Run simulations
    results = main()


