import numpy as np
import pandas as pd

def montecarlo(log_returns, alpha = 0.01, window = 500, days=10, simulaciones=10_000):

    var_list = []
    cvar_list = []
    dates = []

    # for every time t
    for t in range(len(log_returns)):

        montecarlo = []
        
        # not enough data yet
        if t + 1 < window:
            var_list.append(np.nan)
            cvar_list.append(np.nan)
            dates.append(log_returns.index[t])
            continue
        
        # 2. Se construye una base cod 501 observaciones
        sample = log_returns.iloc[t + 1 - window : t + 1].values  # last `window` log_returns including today

        # 5. media y varianza de los log_returns y se anualizan
        m = np.mean(sample) 
        s = np.std(sample) 

        # 6. se estima un retorno random para manana 10_000 veces!
        
        # for simulacion in range(simulaciones):
        #     # e es un cuantil random de la normal 
        #     e = np.random.normal(0, 1)
        #     montecarlo.append(np.exp(m + s * e) - 1)

        e = np.random.normal(0, 1, size=simulaciones)
        montecarlo = np.exp(m + s * e) - 1


        montecarlo_sorted = np.sort(montecarlo)                        # ascending: worst log_returns first

        k = int(np.ceil(alpha * simulaciones))                       # number of tail observations
        k = max(k, 1)                                          # safety: at least 1

        # VaR threshold return is the k-th worst return (0-indexed -> k-1)
        q_alpha = montecarlo_sorted[k - 1]                         # (approximately) alpha-quantile

        VaR = -q_alpha                                         # positive loss number
        CVaR = -montecarlo_sorted[:k - 1].mean()                       # mean of worst k log_returns (positive loss)

        var_list.append(VaR)
        cvar_list.append(CVaR)
        dates.append(log_returns.index[t])
    
    print(f"Para las {window} observaciones, Para un VaR al {(1-alpha)*100}% de confianza se usa la posicion {k} para el VaR, con el promedio de {k-1} para el CVaR.")
    
    var_series = pd.Series(var_list, index=dates, name=f"VaR {int((1-alpha)*100)}% 1d")
    cvar_series = pd.Series(cvar_list, index=dates, name=f"CVaR {int((1-alpha)*100)}% 1d")
    var_10d_series = pd.Series(var_series * np.sqrt(days), index=dates, name=f"VaR {int((1-alpha)*100)}% {days}d")
    capital_series = pd.Series(3 * var_10d_series, index=dates, name=f"Capital {int((1-alpha)*100)}% {days}d")

    df = log_returns.copy()
    df[var_series.name] = var_series
    df[cvar_series.name] = cvar_series
    df[var_10d_series.name] = var_10d_series
    df[capital_series.name] = capital_series

    df = pd.DataFrame({
                     "log_returns": log_returns,
                     var_series.name: var_series,
                     cvar_series.name: cvar_series,
                     var_10d_series.name: var_10d_series,
                     capital_series.name: capital_series})

    # drop initial nans
    df = df.dropna()

    return df

