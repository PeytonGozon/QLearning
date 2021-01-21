# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%
NUM_BATCHES = 10 
m = 75_000    # m
l = 50_000     # l
FIRST_EPISODE = 500
FULL_EXPLOIT_RATIO = 0.75

GAME_NAME = 'Tic Tac Toe'

X_AXIS = "Episode"
Y_AXIS = "Win Rate (%)"
TITLE = "Q-Learning AI win rate playing {} vs Random AI".format(GAME_NAME)

fill_region_x = [l, m]
fill_region_explore_and_exploit = [0, l]


x = [FIRST_EPISODE]
for i in range(NUM_BATCHES+1):
    x.append((i+1) * (m / NUM_BATCHES))

fig, ax = plt.subplots()

ax.fill_between(fill_region_x, 0, 100, alpha=0.75, label='fully exploit')
ax.fill_between(fill_region_explore_and_exploit, 0, 100, alpha=0.25, label='explore and exploit')
plt.ylim(0, 100)
plt.yticks(np.arange(0, 100, 5))
plt.xlabel(X_AXIS)
plt.ylabel(Y_AXIS)
plt.title(TITLE)

file_name_to_label = {
    'tictactoe.csv': ('--bo', 'Tic Tac Toe'), 
    'tictactoe4.csv': ('--rx', 'Tic Tac Toe (4x4)'),
    'tictactoe5.csv': ('--gv', 'Tic Tac Toe (5x5)')
}

# for file_name in ['tictactoe.csv', 'tictactoe4.csv', 'tictactoe5.csv']:
data = pd.read_csv('CSVs/Hex-75000-alpha0.1-gamma0.9-a0.5-b0.0-l50000.csv')
data.head()

# Perform summary statistics
# data = data.drop('Batch 200000', axis=1)
data = data.transpose()

# Get the average of each row
data['mean'] = data.mean(axis=1)
data['winrate'] = data['mean'] * 100
# create data

# plotting_info = file_name_to_label[file_name]
ax.plot(x, data['winrate'], '--bo')

# ax.plot(x, data['winrate'], plotting_info[0], label=plotting_info[1])

plt.legend(loc='lower left')
plt.savefig(GAME_NAME+'.png')
plt.show()
%%

# %%
