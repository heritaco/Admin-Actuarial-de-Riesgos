import numpy as np
import pandas as pd

def montecarlo_ndias(log_returns, alpha = 0.01, window = 500, days=10, simulaciones=10_000, days_to_simulate = 252):

    var_list = []
    cvar_list = []
    dates = []

    # for every time t
    for t in range(len(log_returns)):
       
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
        T = days_to_simulate

        # 6. se estima un retorno random para manana 10_000 veces!
        e = np.random.normal(0, 1, size=simulaciones)
        returns_T = np.exp(T*m + np.sqrt(T)*s*e) - 1

        returns_T.sort()

        k = int(np.ceil(alpha * simulaciones)) # number of tail observations
        k = max(k, 1)                                          # safety: at least 1

        # VaR threshold return is the k-th worst return (0-indexed -> k-1)
        q_alpha = returns_T[k - 1]                         # (approximately) alpha-quantile

        VaR = -q_alpha                                         # positive loss number
        CVaR = -returns_T[:k - 1].mean()                       # mean of worst k log_returns (positive loss)

        var_list.append(VaR)
        cvar_list.append(CVaR)
        dates.append(log_returns.index[t])
    
    print(f"Para {window} retornos, para un VaR al {(1-alpha)*100}% de confianza, para {simulaciones} simulaciones a {days_to_simulate} días para simular, se usa la posición {k} para el VaR, con el promedio de {k-1} para el CVaR.")
    
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

