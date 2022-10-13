import sys
import time
import pickle
import random
import gc
import numpy as np
import tensorflow.keras.backend as K
from sklearn import preprocessing

from log import Logger
from cvaest_architecture import *
from plot_history import plot_total_loss, plot_kl_loss, plot_reconstruction_loss

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def shuffle_data(profiling_x, label_y, plt, key):
    """
    :param profiling_x: input traces
    :param label_y: targeted variable related to each trace
    :param plt: plaintext related to each trace
    :param key: secret key to shuffle related to each trace
    :return: shuffled datasets with the same permutation
    """

    l = list(zip(profiling_x,label_y, plt, key))
    random.shuffle(l)
    shuffled_x, shuffled_y, shuffled_plt, shuffled_key = list(zip(*l))
    return (np.array(shuffled_x), np.array(shuffled_y), np.array(shuffled_plt), np.array(shuffled_key))

# Implementation based on:
# "Sylvain Guilley, Annelie Heuser, Ming Tang, Olivier Rioul. Stochastic Side-Channel Leakage Analysis via Orthonormal Decomposition.
# Int. Conf. Information Technology and Communications Security (SECITC 2017), Jun 2017, Bucharest, Romania."
def fast_monomial_orthonormal_basis(X_traces, X_plaintext, X_key, base_u):
    """
    :param X_traces: input traces
    :param X_plaintext: plaintext related to each trace
    :param X_key: secret key related to each trace
    :param base_u: basis of the monomial subspace (F(1) => base_u=1 / F(2) => base_u=9 / F(3) => base_u=38 / F(4) => base_u=94 / F(5) => base_u=164 / F(6) => base_u=220 / F(7) => base_u=248 / F(8) => base_u=256 / F(9) => base_u=257)
    :return: Projection of the input traces onto the monomial orthonormal basis
    """

    # Computation of the constant term in the Fourier basis
    nb_bit_max = 8 # Number of bits which characterizes X_plaintext and X_key
    nb_hv = 2**nb_bit_max # Number of classes
    num = 2**(-nb_bit_max/2)

    basis_vec = np.arange(nb_hv)
    u = ['{:08b}'.format(i) for i in basis_vec] # basis vector

    # Computation of the targeted variable
    phi_multivariate = np.zeros(X_traces.shape[0])
    for i in range(X_traces.shape[0]):
        phi_multivariate[i] = AES_Sbox[X_plaintext[i]^X_key[i]]
        
    # Initialization of the matrices
    Xphi_traces = np.zeros((X_traces.shape[0], X_traces.shape[1], nb_hv))
    Xphi_traces_sorted = np.zeros((X_traces.shape[0], X_traces.shape[1], nb_hv))
    G = np.zeros((nb_hv, nb_hv))

    # Basis computation (see Theorem 5 of the paper mentioned in the preamble)
    for i in range(nb_hv):
        for j in range(nb_hv):
            phi_proj = 1
            for b in range(nb_bit_max):
                phi_proj *= (1 - 2*int(bin(i)[2:].zfill(nb_bit_max)[b]))**(int(u[j][b]))
            G[i,j] = phi_proj 

    # Trace projection onto the monomial orthonormal basis (see Eq.7 of the paper mentioned in the preamble)
    for i in range(Xphi_traces.shape[0]):
        for j in range(Xphi_traces.shape[1]):
            Xphi_traces[i,j,:] = X_traces[i,j] * (num * G[int(phi_multivariate[i]),:])

        # Ordering the projection to ease the interpretation process
        hyp_values = np.zeros(nb_hv)
        for j in range(nb_hv):
            hyp_values[j] = bin(j)[2:].count('1')

        counter1, counter2, counter3, counter4, counter5, counter6, counter7, counter8 = 0, 0, 0, 0, 0, 0, 0, 0
        for j in range(nb_hv):
            if j == 0:
                Xphi_traces_sorted[i,:,0] = Xphi_traces[i,:,0]

            if hyp_values[j]==1:
                Xphi_traces_sorted[i,:, 1 + counter1] = Xphi_traces[i,:,j]
                counter1+=1

            elif hyp_values[j]==2:
                Xphi_traces_sorted[i,:, 9 + counter2] = Xphi_traces[i,:,j]
                counter2+=1

            elif hyp_values[j]==3:
                Xphi_traces_sorted[i,:, 37 + counter3] = Xphi_traces[i,:,j]
                counter3+=1

            elif hyp_values[j]==4:
                Xphi_traces_sorted[i,:, 93 + counter4] = Xphi_traces[i,:,j]
                counter4+=1

            elif hyp_values[j]==5:
                Xphi_traces_sorted[i,:, 163 + counter5] = Xphi_traces[i,:,j]
                counter5+=1

            elif hyp_values[j]==6:
                Xphi_traces_sorted[i,:, 219 + counter6] = Xphi_traces[i,:,j]
                counter6+=1

            elif hyp_values[j]==7:
                Xphi_traces_sorted[i,:, 247 + counter7] = Xphi_traces[i,:,j]
                counter7+=1

            elif hyp_values[j]==8:
                Xphi_traces_sorted[i,:, 255 + counter8] = Xphi_traces[i,:,j]
                counter8+=1

    return Xphi_traces_sorted[:,:,:base_u]

#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root="./"
trained_models_folder = root+"trained_models/"
history_folder = root+"training_history/"

# Settings & Hyperparameters
nb_epochs = 50
batch_size = 32
learning_rate = 0.01
targeted_byte = 0
base_u = 257 # (F(1) => base_u=1 / F(2) => base_u=9 / F(3) => base_u=38 / F(4) => base_u=94 / F(5) => base_u=164 / F(6) => base_u=220 / F(7) => base_u=248 / F(8) => base_u=256 / F(9) => base_u=257)
scenario="scenario4"
noise = "low_noise"

input_size = 3
train_data_len = 9000
valid_data_len = 1000

model_name = "cVAESA_lr"+str(learning_rate)+"_proftraces"+str(train_data_len)+"_"+str(noise)+'_'+str(scenario)

# Log Initialization
sys.stdout = Logger(model_name)

print('\n############### Configuration ###############\n')

print('Model name = '+model_name)
print('loss = ELBO Loss')
print('nb_epochs = ', nb_epochs)
print('batch_size = ', batch_size)
print('learning_rate = ', learning_rate)
print('targeted_byte = ', targeted_byte)
print('base_u = ', base_u)
print('input_size = ', input_size)
print('train_data_len = ', train_data_len)
print('valid_data_len = ', valid_data_len)
print(scenario)
print(noise)

print('\n############### Data ###############\n')
traces = np.load(root+'simulated_traces/'+str(scenario)+'/traces/traces_'+str(scenario)+'_'+str(noise)+'.npy')
plaintext = np.load(root+'simulated_traces/'+str(scenario)+'/plaintext/plaintext_'+str(scenario)+'_'+str(noise)+'.npy')
intermediate_values = np.load(root+'simulated_traces/'+str(scenario)+'/intermediate_value/intermediate_value_'+str(scenario)+'_'+str(noise)+'.npy')
key = np.load(root+'simulated_traces/'+str(scenario)+'/key/key_'+str(scenario)+'_'+str(noise)+'.npy')
print('Data Loaded ! \n')

# Profiling/Validation data
X = traces[:train_data_len+valid_data_len,:]
Y = intermediate_values[:train_data_len+valid_data_len,targeted_byte]
plaintext = plaintext[:train_data_len+valid_data_len,targeted_byte]
key = key[:train_data_len+valid_data_len,targeted_byte]

# Preprocessing: Shuffling + Normalization (zero mean, unit variance)
(X, Y, plaintext, key) = shuffle_data(X, Y, plaintext, key)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X = X.astype('float32')

# Decomposition into Profiling vs. Validation set
X_profiling = X[:train_data_len,:]
Y_profiling = Y[:train_data_len]
plaintext_profiling = plaintext[:train_data_len]
key_profiling = key[:train_data_len]
print('Traces Profiling loaded !')

X_val = X[train_data_len:train_data_len+valid_data_len,:]
Y_val = Y[train_data_len:train_data_len+valid_data_len]
plaintext_val = plaintext[train_data_len:train_data_len+valid_data_len]
key_val = key[train_data_len:train_data_len+valid_data_len]
print('Traces Validation loaded !')

#################################################
#################################################

####             Training Stage            ######

#################################################
#################################################

# Configuration of the encoder
encoder = define_encoder(input_size=input_size, base_u=base_u-1)
print(encoder.summary())
keras.utils.plot_model(encoder, to_file='architecture_encoder.png', show_shapes=True, show_layer_names=True)
print('\n')

# Configuration of the discriminator
decoder = define_decoder(input_size=input_size, base_u=base_u-1)
print(decoder.summary())
keras.utils.plot_model(decoder, to_file='architecture_decoder.png', show_shapes=True, show_layer_names=True)
print('\n')

# Initiatilization/Compilation of the cVAE-SA
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(lr=learning_rate))

print("\n############### Starting Training #################\n")
start = time.time()

# Initialization of the conditional input of the cVAE-SA
empty_profiling_basis = fast_monomial_orthonormal_basis(np.ones((X_profiling.shape[0], X_profiling.shape[1])), plaintext_profiling, key_profiling, base_u-1)
empty_val_basis = fast_monomial_orthonormal_basis(np.ones((X_val.shape[0], X_val.shape[1])), plaintext_val, key_val, base_u-1)

history = vae.fit([X_profiling, empty_profiling_basis], validation_data = ([X_val, empty_val_basis]), epochs=nb_epochs, batch_size=batch_size)

end=time.time()
print('Overlap time = %d'%(end-start))

# Save/Plot the metrics' evolution
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
plot_total_loss(history_folder + 'history_' + model_name, 'loss_'+model_name)
plot_kl_loss(history_folder + 'history_' + model_name, 'loss_'+model_name)
plot_reconstruction_loss(history_folder + 'history_' + model_name, 'loss_'+model_name)

# Save Decoder/Encoder
filename1 = trained_models_folder+'decoder'
decoder.save(filename1)
filename2 = trained_models_folder+'encoder'
encoder.save(filename2)

print("\n############### Training Done #################\n")

K.clear_session()
gc.collect()