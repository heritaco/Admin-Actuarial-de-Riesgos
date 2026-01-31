import numpy as np
import pandas as pd


def historica(returns, alpha=0.01, window=500, days=10):

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

        # 2. Se construye una base con 500 retornos
        sample = returns.iloc[t + 1 - window: t + 1].values

        # Se ordenan de mas menor a mayor
        sample_sorted = np.sort(sample)

        # Numero de observaciones (son 5 para el 99% de 500 retornos)
        k = int(np.ceil(alpha * window))
        k = max(k, 1)

        # VaR es la 5a observacion, CVaR es el promedio de las primeras 4
        VaR = -sample_sorted[k - 1]
        CVaR = -sample_sorted[:k - 1].mean()

        var_list.append(VaR)
        cvar_list.append(CVaR)
        dates.append(returns.index[t])

    print(f"Para las {window} observaciones, Para un VaR al {(1-alpha)*100}% de confianza se usa la posicion {k} para el VaR, con el promedio de {k-1} para el CVaR.")

    var_series = pd.Series(var_list, index=dates,
                           name=f"VaR {int((1-alpha)*100)}% 1d")
    cvar_series = pd.Series(cvar_list, index=dates,
                            name=f"CVaR {int((1-alpha)*100)}% 1d")
    var_10d_series = pd.Series(
        var_series * np.sqrt(days), index=dates, name=f"VaR {int((1-alpha)*100)}% {days}d")
    capital_series = pd.Series(
        3 * var_10d_series, index=dates, name=f"Capital {int((1-alpha)*100)}% {days}d")

    df = pd.DataFrame({
        "Returns": returns,
        var_series.name: var_series,
        cvar_series.name: cvar_series,
        var_10d_series.name: var_10d_series,
        capital_series.name: capital_series})
    df = df.dropna()

    return df
