import numpy as np
import pandas as pd

def VaR_CVaR_Simulacion_Historica(returns, alpha = 0.01, window = 500, days=10):

    var_list = []
    cvar_list = []
    dates = []

    # for every time t
    for t in range(len(returns)):
        
        # not enough data yet
        if t + 1 < window:
            var_list.append(np.nan)
            cvar_list.append(np.nan)
            dates.append(returns.index[t])
            continue
        
        sample = returns.iloc[t + 1 - window : t + 1].values  # last `window` returns including today
        sample_sorted = np.sort(sample)                        # ascending: worst returns first

        k = int(np.ceil(alpha * window))                       # number of tail observations
        k = max(k, 1)                                          # safety: at least 1

        # VaR threshold return is the k-th worst return (0-indexed -> k-1)
        q_alpha = sample_sorted[k - 1]                         # (approximately) alpha-quantile

        VaR = -q_alpha                                         # positive loss number
        CVaR = -sample_sorted[:k - 1].mean()                       # mean of worst k returns (positive loss)

        var_list.append(VaR)
        cvar_list.append(CVaR)
        dates.append(returns.index[t])
    
    print(f"Para las {window} observaciones, Para un VaR al {(1-alpha)*100}% de confianza se usa la posicion {k} para el VaR, con el promedio de {k-1} para el CVaR.")
    
    var_series = pd.Series(var_list, index=dates, name=f"VaR {int((1-alpha)*100)}% 1d")
    cvar_series = pd.Series(cvar_list, index=dates, name=f"CVaR {int((1-alpha)*100)}% 1d")
    var_10d_series = pd.Series(var_series * np.sqrt(days), index=dates, name=f"VaR {int((1-alpha)*100)}% {days}d")
    capital_series = pd.Series(3 * var_10d_series, index=dates, name=f"Capital {int((1-alpha)*100)}% {days}d")

    df = returns.copy()
    df[var_series.name] = var_series
    df[cvar_series.name] = cvar_series
    df[var_10d_series.name] = var_10d_series
    df[capital_series.name] = capital_series

    df = pd.DataFrame({
                     "Returns": returns,
                     var_series.name: var_series,
                     cvar_series.name: cvar_series,
                     var_10d_series.name: var_10d_series,
                     capital_series.name: capital_series})

    # drop initial nans
    df = df.dropna()

    return df

