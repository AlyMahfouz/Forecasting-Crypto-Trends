# %%
from binance.client import Client
import config

from getHistoricalData import fetch

# %%
client = Client(config.pKey, config.sKey)

# %% 
data = fetch(Client, client, 'BTCUSDT', '17/08/2017', '08/01/2022', '17082017_08012022_1DAY')

# %%
