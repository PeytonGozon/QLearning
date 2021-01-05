# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%
NUM_TIMES_REPORTING = 10 
NUM_EPISODES = 200_000
FIRST_EPISODE = 500
FULL_EXPLOIT_RATIO = 0.75

GAME_NAME = 'Connect 4'

X_AXIS = "Episode"
Y_AXIS = "Win Rate (%)"
TITLE = "Q-Learning AI win rate playing {} vs Random AI".format(GAME_NAME)

fill_region_x = [NUM_EPISODES * FULL_EXPLOIT_RATIO, NUM_EPISODES]
fill_region_explore_and_exploit = [0, NUM_EPISODES * FULL_EXPLOIT_RATIO]


x = [FIRST_EPISODE]
for i in range(NUM_TIMES_REPORTING+1):
    x.append((i+1) * (NUM_EPISODES / NUM_TIMES_REPORTING))

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
data = pd.read_csv('connect4.csv')
data.head()

# Perform summary statistics
# data = data.drop('Batch 200000', axis=1)
data = data.transpose()

# Get the average of each row
data['mean'] = data.mean(axis=1)
data['winrate'] = data['mean'] * 100
# create data

plotting_info = file_name_to_label[file_name]
ax.plot(x, data['winrate'], '--bo')

# ax.plot(x, data['winrate'], plotting_info[0], label=plotting_info[1])

plt.legend(loc='lower left')
plt.savefig(GAME_NAME+'.png')
plt.show()
%%

# %%
