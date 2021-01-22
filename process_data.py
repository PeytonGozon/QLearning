# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%
REPORT_EVERY = 3_000
m = 45_000   
l = 30_000     

GAME_NAME = 'Tic Tac Toe'
ADDITIONAL_INFO = r'with $\alpha = 0.1$'

X_AXIS = "Episode"
Y_AXIS = "Win Rate (%)"
TITLE = 'Q-Agent vs Random AI playing {} {}'.format(GAME_NAME, ADDITIONAL_INFO)

fill_region_x = [l, m]
fill_region_explore_and_exploit = [0, l]


x = [(i+1) * REPORT_EVERY for i in range(1+(m // REPORT_EVERY))]

print(len(x))
print(data.shape)

fig, ax = plt.subplots()

ax.fill_between(fill_region_x, 0, 100, alpha=0.75, label='fully exploit')
ax.fill_between(fill_region_explore_and_exploit, 0, 100, alpha=0.25, label='explore and exploit')

CSVs = [
    ('CSVs/tictactoe-45000-alpha0.1-gamma0.9-a0.5-b0.0-l30000-usingDynamicEps true.csv', '--bo', r'dynamic $\varepsilon'),
    ('CSVs/tictactoe-45000-alpha0.1-gamma0.9-a0.5-b0.0-l30000-usingDynamicEps true.csv', '--bo', r'$\varepsilon = 0.1'),
]

for csv in CSVs:
    data = pd.read_csv(csv[0])
    data = data.transpose()

    # Get the winrate across each row
    data['mean'] = data.mean(axis=1)
    data['winrate'] = data['mean'] * 100

    ax.plot(x, data['winrate'], csv[1], label=csv[2])

plt.ylim(0, 100)
plt.yticks(np.arange(0, 105, 5))

x_ticks = np.concatenate(([0], x[:-1]), axis=None)
plt.xticks(x_ticks, rotation=45)

plt.xlabel(X_AXIS)
plt.ylabel(Y_AXIS)
plt.title(TITLE)

plt.legend(loc='lower right')

plt.savefig(GAME_NAME+'.png')
plt.show()
%%

# %%
