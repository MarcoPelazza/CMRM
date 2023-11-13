#import libraries
import tqdm
from tqdm import tqdm
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#QUESTION 1----------------------------------------------------------

#import database
import deeplake
ds = deeplake.load("hub://activeloop/gtzan-genre", access_method = "stream")

Fs = 22050
n_samples = 29 * Fs # duration in samples 

# Define downsampling factors
sub_train = 10
sub_test = 50

train_range = list(range(0,999,sub_train))
test_range = list(range(0,999,sub_test))

# Extract classes
genre_names = ["pop", "metal", "classical", "rock", "bues", "jazz", "hipkop", "reggae", "disco", "country"]

genre_train = np.take(ds.genre, train_range, axis = 0)
genre_test = np.take(ds.genre, train_range, axis = 0)
print(np.shape(genre_train))
print(np.shape(genre_test))

Fs = 22050
n_samples = 29 * Fs # duration in samples 

# Define downsampling factors
sub_train = 10
sub_test = 50

train_range = list(range(0,999,sub_train))
test_range = list(range(0,999,sub_test))

# Extract classes
genre_names = ["pop", "metal", "classical", "rock", "bues", "jazz", "hipkop", "reggae", "disco", "country"]

genre_train = np.take(ds.genre, train_range, axis = 0)
genre_test = np.take(ds.genre, train_range, axis = 0)
print(np.shape(genre_train))
print(np.shape(genre_test))

# Extract training set
audio_train = []
for i in tqdm(train_range):
    audio_train.append(ds.audio[i, 0:n_samples, 0].numpy(aslist = True))
    print(np.shape(audio_train))

# Extract test set
audio_test = []
for i in tqdm(test_range):
    audio_test.append(ds.audio[i,0:n_samples, 0].numpy(aslist = True))
    print(np.shape(audio_test))

# Plot the first wav in the train set
plt.figure(figsize = (10, 4))
plt.plot(range(n_samples), audio_train[0])
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

#QUESTION 2----------------------------------------------------------

# Preprocessing
scaler = MinMaxScaler(feature_range = (-1,1))
pre_audio_train = scaler.fit_transform(audio_train)
pre_audio_test = scaler.fit_transform(audio_test)

# Plot the first wav in the train set after preprocessing
plt.figure(figsize = (10,4))
plt.plot(range(n_samples), pre_audio_train[0])
plt.show()

# Compute local average
def compute_local_average(x, M):
    """Compute local average of signal

    Args:
        x: Signal
        M: Total length in samples of centric window  used for local average

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M*Fs))
    local_average = np.zeros(L)

    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])

    return local_average

# Compute the principal argument
def principal_argument(x):
    """Principal argument function 
    
    Args:
        x: value (or vector of values)
        
    Returns:
        y: Principal value of x
    """
    L = len(x)
    y = np.zeros(L)
    for i in range(L):
        y = [((n + 0.5) % 1) - 0.5 for n in x]
    return y

# Compute the Phase-Based Novelty function
def compute_phase_novelty(x, Fs=1, N=1024, H=64, M=40, norm=True, plot=False):
    """Compute phase-based novelty function

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hop size
        M: Total length in samples of centric window  used for local average
        norm: Apply max norm (if norm==True)
        plot: plot novelty (if plot==True)

    Returns:
        nov: Phase-based novelty function
        Fs_nov: Novelty rate
    """
    
    # Compute the STFT
    X = librosa.stft(x,n_fft = N, hop_length = H, win_length = N, window = 'hann')
    
    # Compute the novelty rate
    Fs_nov = Fs/H
    
    # Extract the phase and use principal_argument
    phase = np.angle(X)/(2*np.pi)
    phase_diff = principal_argument(np.diff(phase, axis = 1))
    phase_diff2 = principal_argument(np.diff(phase_diff, axis = 1))
    # Accumulation over frequency axis
    nov = np.sum(np.abs(phase_diff2), axis = 0)
    nov = np.concatenate((nov, np.array([0,0])))
    
    # Local average subtraction and half-wave rectification
    if M > 0:
        local_average = compute_local_average(nov,M)
        nov = nov - local_average
        nov[nov<0] = 0
    
    # Normalization
    if norm: 
        max = np.max(nov)
        nov = nov/max
    
    # Plot
    if plot:
        feature_time_axis = np.arange(nov.shape[0])
        plt.figure(figsize=(10,4))
        plt.plot(feature_time_axis, nov)
    
    return nov, Fs_nov

# Test the novelty function on the first wav in the train set

compute_phase_novelty(pre_audio_train[0], plot = True)





