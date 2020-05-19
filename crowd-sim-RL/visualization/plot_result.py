import json
import matplotlib.pyplot as plt
from pandas import np
from scipy.signal import savgol_filter


def main():
    plot_reward_progress()

    #plot_comparison()

    #plot_cap()

    #plot_values()


def plot_comparison():
    path = ""

    avg_ddpg = []
    avg_ppo_1 = []
    avg_ppo_8 = []

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(avg_ddpg))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth * 2 for x in r1]

    # Make the plot
    plt.bar(r1, avg_ddpg, color='#f59642', width=barWidth, edgecolor='white', label='DDPG - 1 worker')
    plt.bar(r2, avg_ppo_1, color='#4266f5', width=barWidth, edgecolor='white', label='PPO - 1 worker')
    plt.bar(r3, avg_ppo_8, color='#42bcf5', width=barWidth, edgecolor='white', label='PPO - 7 workers')

    # Add xticks on the middle of the group bars
    plt.title("Hallway 2 pedestrians per side", fontweight="bold")
    #plt.xticks([r + barWidth for r in range(len(avg_times_1))], ['Centralized', 'Concurrent', 'PS', 'Central critic'])
    plt.xticks([r + barWidth for r in range(len(avg_ddpg))], ['Concurrent', 'PS', 'Central critic'])
    plt.ylabel('Time (s)', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def plot_cap():
    barWidth = 0.25

    cap = []
    nocap = []

    # Set position of bar on X axis
    r1 = np.arange(len(cap))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, cap, color='#4266f5', width=barWidth, edgecolor='white', label='Laser cap = 10')
    plt.bar(r2, nocap, color='#42bcf5', width=barWidth, edgecolor='white', label='No laser cap')

    # Add xticks on the middle of the group bars
    # plt.xlabel('3-confusion', fontweight='bold')
    plt.title("Capping lasers vs. leaving them uncapped", fontweight="bold")
    plt.xticks([r + barWidth / 2 for r in range(len(cap))], ['PS hallway', 'Conc. hallway', 'PS cross 3v3', 'Conc. cross 3v3'])
    plt.ylabel('Time (s)', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def plot_values():
    barWidth = 0.50

    avg = []

    # Set position of bar on X axis
    r1 = np.arange(len(avg))

    # Make the plot
    plt.bar(r1, avg, color='#42bcf5', width=barWidth, edgecolor='white', label='PPO 7 workers')

    # Add xticks on the middle of the group bars
    plt.title("Crossway 5v5", fontweight="bold")
    plt.xticks([r for r in range(len(avg))], ['Concurrent', 'PS', 'Central critic'])
    plt.ylabel('Time (s)', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def plot_reward_progress():
    files = []
    path = ""

    dicts = []
    for file in files:
        file_path = path + file + "/result.json"
        with open(file_path, 'r') as handle:
            json_data = [json.loads(line) for line in handle]

        dict = {}
        for item in json_data:
            dict[item["time_total_s"]] = item["episode_reward_mean"]
            if item["episode_reward_mean"] > 1140:
                break
        dicts.append(dict)

    plot(dicts)


def plot(data):

    plt.figure()

    legends = ["Concurrent 7 workers", "PS 7 workers", "Central Critic 1 worker"]

    i = 0
    for d in data:
        lists = sorted(d.items())
        x, y = zip(*lists)
        y = savgol_filter(y, 11, 4)

        plt.title("Crossway 5v5", fontweight="bold")
        plt.xlabel("Time (s)", fontweight="bold")
        plt.ylabel("Mean episode reward", fontweight="bold")
        plt.plot(x, y)
        plt.ylim(bottom=700)
        i += 1
    plt.axhline(y=1140, color='r')

    plt.legend(legends, loc=4, fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
