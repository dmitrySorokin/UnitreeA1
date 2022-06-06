from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot(x, y, ylow, yup, xlabel, ylabel, name):
    plt.plot(x, y)
    plt.fill_between(x, ylow, yup, alpha=0.5)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=1.08)

    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(
        open('results.csv', 'r'),
        sep=';',
        header=None,
        index_col=None,
        names=['target_speed', 'reward_sum', 'reward_sum_std', 'velocity_rew', 'velocity_rew_std', 'posx', 'posx_std', 'posy', 'posy_std', 'speedx', 'speedx_std', 'speedy', 'speedy_std']
    )
    df = df.sort_values(by='target_speed')

    target_speed = df['target_speed'].to_numpy()
    speedx = df['speedx'].to_numpy()
    speedx_std = df['speedx_std'].to_numpy()
    speedy = df['speedy'].to_numpy()
    speedy_std = df['speedy_std'].to_numpy()

    posx = df['posx'].to_numpy()
    posx_std = df['posx_std'].to_numpy()

    posy = df['posy'].to_numpy()
    posy_std = df['posy_std'].to_numpy()

    tot_rew = df['reward_sum'].to_numpy()
    tot_rew_std = df['reward_sum_std'].to_numpy()
    vel_rew = df['velocity_rew'].to_numpy()
    vel_rew_std = df['velocity_rew_std'].to_numpy()

    plot(target_speed, tot_rew, tot_rew - tot_rew_std, tot_rew + tot_rew_std, '$V_{target}, m/s$', 'tot_rew', 'tot.png')
    plot(target_speed, vel_rew, vel_rew - vel_rew_std, vel_rew + vel_rew_std, '$V_{target}, m/s$', 'vel_rew', 'vel.png')

    plot(target_speed, posx, posx - posx_std, posx + posx_std, '$V_{target}, m/s$', '$d_x$, m', 'distx.png')
    plot(target_speed, posy, posy - posy_std, posy + posy_std, '$V_{target}, m/s$', '$d_y$, m', 'disty.png')

    plot(target_speed, speedx, speedx - speedx_std, speedx + speedx_std, '$V_{target}, m/s$', '$V_x$, m/s', 'speedx.png')
    plot(target_speed, speedy, speedy - speedy_std, speedy + speedy_std, '$V_{target}, m/s$', '$V_y$ m/s', 'speedy.png')

    plot(
        target_speed,
        (speedx - target_speed) / target_speed,
        (speedx - speedx_std - target_speed) / target_speed,
        (speedx + speedx_std - target_speed) / target_speed,
        '$V_{target}, m/s$',
        '$\\varepsilon_x$',
        'errorx.png'
    )
