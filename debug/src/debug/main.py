from arfima_model import ARFIMA
import numpy as np
import json
import load_data
import common
from graphs import visualize_time_series


symbol = "BTCUSDT"
interval = "1h"
since = "2026-11-01T00:00:00Z"
output = "price_series.json"
split_size = 0.70
horizon = 100

common.file_clean(path="price_series.json")
print("File cleaned")
load_data.get_price_series(symbol, interval, since, output)


with open('./price_series.json', 'r') as f:
    input_data = np.array(json.load(f))

cut = int(len(input_data) * split_size)
train, test = input_data[:cut], input_data[cut:]

freq = 7
alpha = 0.05
ub = 0.499
hybrid = True
min_window = 50
rs_iters = 2000
max_box_cox_iters = 200
nlags = 20
max_iters_optim = 50
optimizer = "aic"


m = ARFIMA(
    freq,
    alpha,
    ub,
    hybrid,
    max_box_cox_iters,
    min_window,
    rs_iters,
    nlags,
    max_iters_optim,
    optimizer
)
m.fit(train)

pred = m.forecast(train, horizon)

visualize_time_series(input_data, pred, split_size, title="Rust ARFIMA")
