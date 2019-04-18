import numpy as np, scipy, pandas as pd, seaborn as sns, torch
from matplotlib import pyplot as plt
from torch.utils import data

sns.set()

def compute_cross_correlation(TS, center=False, window=30):
    if center is True:
        mean = TS.mean(axis=1)
    else:
        mean = np.zeros(TS.shape[0])
        
    return (TS - mean).dot((TS - mean).T) / (window - 1)

def get_cross_correlation_list(TS, window_size=30, center=False):
    if center is True:
        def compute_cross_correlation(TS_window):
            mean = TS_window.mean(axis=1)
            return (TS_window - mean[:, None]).dot((TS_window - mean[:, None]).T) / (window_size - 1)
    else:
        def compute_cross_correlation(TS_window):
            return TS_window.dot(TS_window.T) / (window_size - 1)
    
    TS_list = [TS[:, i - window_size: i] for i in range(window_size, TS.shape[1])]
    return np.array(list(map(compute_cross_correlation, TS_list)))

def generate_ts(array_len, num_ts, low_freq=40, high_freq=50, low_phase=50, high_phase=75, noise_scale=0.3):
    res = []
    num_array = np.arange(array_len)
    W = np.random.uniform(low=low_freq, high=high_freq, size=num_ts)
    T = np.random.uniform(low=low_phase, high=high_phase, size=num_ts)
    S_rand = np.random.randint(0, high=2, size=num_ts)
    for i in range(num_ts):
        func = np.cos if S_rand[i] == 1 else np.sin
        noise = np.random.standard_normal(array_len)
        res.append(func((num_array - T[i]) / W[i]) + noise_scale * noise)
    return np.array(res)

def inject_shock_wave_anomalies(TS, start_from, num_anomalies=5, magnitude=2, anomaly_len_low=50, anomaly_len_high=200):
    #get random indices starting for anomalies to star at starting at start_from
    random_indices = np.random.permutation(np.array(range(start_from, TS.shape[1] - anomaly_len_high)))[:num_anomalies]
    #get random lengths of for anomalies
    random_length = np.random.randint(anomaly_len_low, anomaly_len_high, size=num_anomalies)
    #get amount of correlated anomalies for each starting point
    number_anomalies = np.random.randint(1, 4, size=num_anomalies)
    #get indices of Time Series with correlated anomalies
    random_ts = [np.random.permutation(np.arange(TS.shape[0]))[:num_an] for num_an in number_anomalies]
    #generate anomalies signs
    sign = [np.where(np.random.randint(0, 2, size=num_an) == 0, -1, 1) for num_an in number_anomalies]
    magnitude_list = [np.array(magnitude) + 0.3 * np.random.standard_normal(size=num_an) for num_an in number_anomalies]
    TS_anom = TS.copy()
    for i in range(num_anomalies):
        TS_anom[random_ts[i], random_indices[i]:random_indices[i] + random_length[i]] += (sign[i] * magnitude_list[i]).reshape(-1, 1)
    return TS_anom, random_ts, random_indices, random_length
    
def generate_tensor_dataset(cross_cor, window_size=5, test=False):
    object_list = []
    target_list = []
    
    if test is True:
        step = 1
        total_len = len(cross_cor) - window_size
    else:
        step = window_size + 1
        total_len = len(cross_cor) - (len(cross_cor) % 6) 
    
    for i in range(0, total_len, step):
        object_list.append(torch.FloatTensor(cross_cor[i: i + window_size, :, :]))
        target_list.append(torch.FloatTensor(cross_cor[i + window_size, :, :]))
        
    return data.TensorDataset(torch.stack(object_list), torch.stack(target_list))

#Survival analysis stuff

def compute_hazard(X, timeline, coeffs, max_duration=100000, rho=1, baseline=None):
    if baseline is None:
        def baseline(timeline):
            return timeline / max_duration ** rho
    return baseline(timeline) * np.exp(np.dot(coeffs, X))

def get_death_time(X, hazard_func, max_obs):
    probs = hazard_func(X)
    generated_probs = np.random.uniform(size=probs.shape)
    ind = np.argwhere(probs > generated_probs)
    if ind.shape[0] == 0:
        return max_obs
    else:
        return ind[0][0]
    
def get_survival_data(num_series, num_var, max_obs, coeffs, rho=1, **kwargs):
    data_list = []
    death_time_list = []
    timeline = np.arange(max_obs)
    for i in range(num_series):
        TS_list = generate_ts(max_obs, num_var)
        TS_list, ts_ind, anomaly_ind, anomaly_len = inject_shock_wave_anomalies(TS=TS_list, **kwargs)
        
        anomaly_counter = np.zeros(TS_list.shape[1])
        for anom_start, anom_len in zip(anomaly_ind, anomaly_len):
            anomaly_counter[anom_start + anom_len:] += 1

        TS_list = np.vstack((TS_list, anomaly_counter))
        hf = lambda X: compute_hazard(X, timeline=timeline, coeffs=coeffs, rho=rho)
        death_time = get_death_time(TS_list, hazard_func=hf, max_obs=max_obs)
        death_time_list.append(death_time)
        data_list.append(TS_list[:, :death_time])
    return data_list, death_time_list

def transform_to_long(data_list, max_obs):
    df_list = []
    columns = list(map(str, np.arange(1, data_list[0].shape[0]))) + ['anomaly_counter']
    for i, ds in enumerate(data_list):
        df = pd.DataFrame(ds.T, columns=columns)
        start = pd.Series(np.arange(ds.shape[1]), name='start')
        stop = pd.Series(np.arange(ds.shape[1]) + 1, name='stop')
        event = pd.Series(np.zeros(ds.shape[1]), name='event')
        id = pd.Series(np.ones(ds.shape[1]).astype(np.int) * (i + 1), name='id')
        if ds.shape[1] < max_obs:
            event.iloc[-1] = 1
        df_list.append(pd.concat((id, start, stop, df, event), axis=1))
    return pd.concat(df_list, axis=0)