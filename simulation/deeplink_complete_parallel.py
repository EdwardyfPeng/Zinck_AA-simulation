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
q=0.2 #FDR

X -= np.mean(X, axis=0)
# Prevent division by zero: Replace zero std with 1 before dividing
std_dev = np.std(X, axis=0, ddof=1)
std_dev[std_dev == 0] = 1  # Avoid division by zero
X /= std_dev

# Normalize X1 while ensuring no division by zero in normalization
norms = np.sqrt(np.sum(X ** 2, axis=0))
norms[norms == 0] = 1  # Avoid division by zero
X1 = X / norms
r_hat = PCp1(X1, 15)

############## Binary Outcomes ################
mlp_loss = 'binary_crossentropy'
Xnew = dl.knockoff_construct(X, r_hat, 'relu', aut_epoch, aut_loss, aut_verb)

p = Xnew.shape[1] // 2
 # implement DeepPINK
es = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
dp = Sequential()
dp.add(PairwiseConnected(input_shape=(2 * p,)))
dp.add(Dense(p, activation='elu', kernel_regularizer=keras.regularizers.l1(l1=l1)))
dp.add(Dense(1, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
dp.compile(loss=mlp_loss, optimizer=keras.optimizers.Adam(learning_rate=lr))
dp.fit(Xnew, Y, epochs=mlp_epoch, batch_size=32, verbose=mlp_verb, validation_split=0.1, callbacks=[es])

 # calculate knockoff statistics W_j
weights = dp.get_weights()
w = weights[1] @ weights[3]
w = w.reshape(p, )
z = weights[0][:p]
z_tilde = weights[0][p:]
W = (w * z) ** 2 - (w * z_tilde) ** 2
# feature selection
selected = dl.knockoff_select(W, q, ko_plus=False)
