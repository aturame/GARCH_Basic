import arch
from arch import arch_model
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import numpy as np


df = pd.read_excel("/Users/atulramesh/Desktop/data2.xlsx")
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
df.set_index('Dates', inplace=True)

df['Returns'] = df['PX_OPEN'].pct_change()
df.dropna(inplace=True)

best_aic = float("inf")
best_pq = None

for p, q in itertools.product(range(1, 3), range(1, 3)):  
    model = arch_model(df['Returns'], p=p, q=q, mean="AR", vol="GARCH", dist="t")
    result = model.fit(disp="off")
    if result.aic < best_aic:
        best_aic = result.aic
        best_pq = (p, q)

print(f"Best GARCH Order: {best_pq} with AIC: {best_aic}")


GARCH_Model = arch_model(df['Returns'], p=best_pq[0], q=best_pq[1], mean="AR", vol="GARCH", dist="t")
gm_result = GARCH_Model.fit(update_freq=1)

fig, axes = plt.subplots(1, 2, figsize=(12,6))
sm.graphics.tsa.plot_acf(gm_result.resid, lags=30, ax=axes[0])
sm.graphics.tsa.plot_pacf(gm_result.resid, lags=30, ax=axes[1])
plt.show()

print(f"Log-Likelihood: {gm_result.loglikelihood}")
print(f"AIC: {gm_result.aic}")
print(f"BIC: {gm_result.bic}")
