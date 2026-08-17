import numpy as np
import properscoring as ps
from tqdm import tqdm 

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CRPS(pred_quantile, true):
    _, Q, _, N = pred_quantile.shape

    pred_quantile = pred_quantile.astype(np.float32)
    true = true.astype(np.float32)

    crps = []
    for i in tqdm(range(N), desc="Calculating CRPS"):
        pred_i = np.transpose(pred_quantile[:, :, :, i], (0,2,1)).reshape(-1, Q).astype(np.float32)
        true_i = true[:, :, i].reshape(-1,).astype(np.float32)
        crps_i = ps.crps_ensemble(true_i, pred_i).mean()
        crps.append(crps_i)
        
    crps = np.mean(crps)
    print("crps: ", crps)
    return crps


def metric(pred, true, pred_q = None):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    if pred_q is not None:
        crps = CRPS(pred_q, true)
        return mae, mse, rmse, mape, mspe, crps
    else:
        return mae, mse, rmse, mape, mspe


def compute_crps_quantile(samples, y, q_levels=None):
    """
    Calculate CRPS based on quantiles.
    
    Parameters:
    samples: Sample output, [samplenum, batchsize, length, targetnum]
    y: Label, [batchsize, length, targetnum]
    q_levels: Selected quantiles, e.g. [0.1, 0.2, ..., 0.9]
    """
    if q_levels is None:
        # Create quantiles list from 0.1 to 0.9 with 0.1 step
        q_levels = np.linspace(0.1, 0.9, 9)
        
    samplenum, batchsize, length, targetnum = samples.shape
    num_quantiles = len(q_levels)

    # Convert samples to quantiles
    quantiles = np.quantile(samples, q_levels, axis=0)  # Shape: [num_quantiles, batchsize, length, targetnum]

    # Compute CRPS based on quantiles
    q_low = q_levels[0]
    q_high = q_levels[-1]
    delta_q = q_levels[1] - q_levels[0]  # Quantile interval (e.g., 0.1)

    # Contribution from the middle region (q₁ to qₙ)
    y_expanded = np.expand_dims(y, axis=0)  # Shape [1, batchsize, length, targetnum]
    indicator = y_expanded <= quantiles  # Shape [num_quantiles, batchsize, length, targetnum]
    q_tensor = np.expand_dims(np.array(q_levels), axis=(1, 2, 3))  # Shape [num_quantiles, 1, 1, 1]
    term_mid = (quantiles - y_expanded) * (indicator.astype(float) - q_tensor)
    contribution_mid = 2 * np.sum(term_mid, axis=0) * delta_q

    # Contribution from the low quantile region (q < q₁)
    Q_low = quantiles[0]
    weight_low = np.where(y <= Q_low, 2 * (q_low - 0.5 * q_low**2), -2 * 0.5 * q_low**2)
    contribution_low = (Q_low - y) * weight_low

    # Contribution from the high quantile region (q > qₙ)
    Q_high = quantiles[-1]
    weight_high = np.where(y <= Q_high, 2 * (0.5 * (1 - q_high)**2), 2 * (-0.5 * (1 - q_high**2) ))
    contribution_high = (Q_high - y) * weight_high

    # Total CRPS
    crps_total = contribution_low + contribution_mid + contribution_high

    return np.mean(crps_total)


def test_crps_quantile(samples, y, q_levels=None):
    """
    Calculate CRPS based on quantiles.
    
    Parameters:
    samples: Sample output, [samplenum, batchsize, length, targetnum]
    y: Label, [batchsize, length, targetnum]
    q_levels: Selected quantiles, e.g. [0.1, 0.2, ..., 0.9]
    """
    if q_levels is None:
        # Create quantiles list from 0.1 to 0.9 with 0.1 step
        q_levels = np.linspace(0.1, 0.9, 9)
        
    samplenum, batchsize, length, targetnum = samples.shape
    num_quantiles = len(q_levels)

    crps_per_seq = []

    for b in range(batchsize):
        y_seq = y[b]                       # shape [length, targetnum]
        samples_seq = samples[:, b]        # shape [samplenum, length, targetnum]

        # abs sum of seq
        abs_target_sum_seq = np.sum(np.abs(y_seq))

        # weighted qloss sum of seq
        weighted_qloss_sum = 0.0
        for q in q_levels:
            q_forecast = np.quantile(samples_seq, q, axis=0)  # [length, targetnum]
            q_loss = 2 * np.abs((q_forecast - y_seq) * ((y_seq <= q_forecast) - q))
            weighted_qloss_sum += np.sum(q_loss) / abs_target_sum_seq

        # CRPS of seq
        crps_seq = weighted_qloss_sum / len(q_levels)
        crps_per_seq.append(crps_seq)
    
    return np.mean(crps_per_seq)