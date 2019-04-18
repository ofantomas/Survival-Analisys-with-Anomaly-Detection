from matplotlib import pyplot as plt
import numpy as np, seaborn as sns, pandas as pd


sns.set()

def draw_mse_scores(scores, label, xlabel="timestamp", ylabel="MSE score", threshold=None, lims=None, title=None, path=None):
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(len(scores)), scores, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([0, len(scores)])
    if title is not None:
        plt.suptitle(title)
    if threshold is not None:
        plt.hlines(threshold, *plt.xlim(), label='Anomaly threshold', linewidths=1, color='r')
    if lims is not None:
        print(lims)
        plt.vlines(lims, 0, scores.max() * 1.1, colors=((1,0,1,1)), label='Anomalous Areas')
    plt.ylim([-scores.max() * 0.05, scores.max() * 1.1])
    plt.legend()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    
def draw_ts(TS, ts_anomaly_list, anomaly_list, anomaly_ind, display_ts=5, offset=150, biased=False, 
            title=None, figsize=(15, 10), seed=None, path=None):
    if seed is not None:
        np.random.seed(seed)
    
    num_ts = TS.shape[0]
    ts_ind = ts_anomaly_list[anomaly_ind]
    num_additional_ts = max(0, display_ts - len(ts_ind))
    ts_ind = ts_anomaly_list[anomaly_ind]

    if num_additional_ts > 0:
        ts_ind = np.concatenate((ts_ind, np.random.permutation(np.setdiff1d(np.arange(num_ts), 
                                                                           ts_ind))[:num_additional_ts]))
    TS_df = TS[ts_ind, (anomaly_list[anomaly_ind] - offset): (anomaly_list[anomaly_ind] + 2 * offset)]
    
    if biased is True:
        TS_df = pd.DataFrame((TS_df + (np.arange(display_ts) * 3)[:, None]).T, columns=ts_ind)
    else:
        TS_df = pd.DataFrame(TS_df.T, columns=ts_ind)
    
    plt.figure(figsize=figsize)
    
    if title is not None:
        plt.title(title)
    
    sns.lineplot(data=TS_df, palette='tab10', linewidth=2, style='choice')
    
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    
    plt.show()


def draw_cross_correlation(cross_cor, title=None, path=None):
    f, ax = plt.subplots(figsize=(10, 8))
    if title is not None:
        f.suptitle(title)
    sns.heatmap(cross_cor, cmap="RdPu", center=0.7,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    
def draw_cross_correlation_difference(cross_cor1, cross_cor2, title=None, path=None):
    f, ax = plt.subplots(figsize=(10, 8))
    if title is not None:
        f.suptitle(title)
    sns.heatmap(cross_cor1 - cross_cor2, cmap="RdPu", center=0.7,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def compute_metrics(y_true, y_predicted):
    tp = ((y_true == 1) * (y_predicted == 1)).sum()
    fp = y_predicted.sum() - tp
    tn = ((y_true == 0) * (y_predicted == 0)).sum()
    fn = y_true.sum() - tp
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def create_anomaly_series(predictions):
    i = 0
    anom_series = np.zeros(len(predictions))
    true_counter = 0
    anom_counter = 0
    while i < len(predictions):
        if predictions[i] == 1:
            while predictions[i] == 1:
                true_counter += 1
                anom_series[i] = anom_counter
                i += 1
            if true_counter >= 5:
                anom_counter += 1
            true_counter = 0
        anom_series[i] = anom_counter
        i += 1
    return anom_series

    