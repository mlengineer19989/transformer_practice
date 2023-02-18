import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def resampling(raw_data, interval):
  resampled_data = []
  for i, v in enumerate(raw_data):
    if i%interval==0:
      resampled_data.append(v)
  return np.array(resampled_data)

def evaluate_naive_method(data):
    total_abs_err = 0.
    preds = [data[0]]
    T = len(data)
    for i in range(T-1):
        pred = data[i]
        target = data[i+1]
        total_abs_err += np.sum(np.abs(pred - target))

        preds.append(pred)
    return total_abs_err / (T-1), np.array(preds)



def calc_score(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return MSE, r2