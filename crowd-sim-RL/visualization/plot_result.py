import glob
import json
import matplotlib.pyplot as plt
from pandas import np
from scipy.signal import savgol_filter


def main():
    #plot_reward_progress()

    plot_comparison()


def plot_comparison():
    path = "/home/lennard/ray_results/ppo_confusion/"

    types = [0, 8]

    schemes = ["central_", "conc_", "ps_", "centralq_"]

    avg_times_1 = []
    avg_times_8 = []

    for type in types:
        for scheme in schemes:
            avg_array = []
            for i in range(1, 5):
                if type == 0 and scheme == "central_" and i > 1 or type == 0 and scheme == "conc_" and i == 5:
                    break
                file_path = path + scheme + str(type) + "_" + str(i) + "/result.json"
                with open(file_path, 'r') as handle:
                    json_data = [json.loads(line) for line in handle]

                time = []
                for item in json_data:
                    time.append(item["time_total_s"])

                avg_array.append(time[-1])

            avg = sum(avg_array) / len(avg_array)

            if type == 0:
                avg_times_1.append(avg)
            else:
                avg_times_8.append(avg)

    # set width of bar
    barWidth = 0.40

    # Set position of bar on X axis
    r1 = np.arange(len(avg_times_1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, avg_times_1, color='#7f6d5f', width=barWidth, edgecolor='white', label='1 worker')
    plt.bar(r2, avg_times_8, color='#557f2d', width=barWidth, edgecolor='white', label='8 workers')

    # Add xticks on the middle of the group bars
    plt.xlabel('3-confusion PPO', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(avg_times_1))], ['Centralized', 'Concurrent', 'PS', 'Central critic'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def plot_reward_progress():
    """phases_path = "/home/lennard/ray_results/ddpg_curr2/"
    outfile = open(phases_path + "result.json", "w")

    for f in glob.glob(phases_path + "*/result.json"):
        with open(f, "r") as infile:
            for obj in infile:
                json.dump(json.loads(obj), outfile)
                outfile.write('\n')"""

    """path = "/home/lennard/ray_results/ddpg_speedup/"
    files = ["ddpg_all", "ddpg_curr"]"""

    path = "/home/lennard/ray_results/ppo_confusion/"
    files = ["central_8", "conc_8", "ps_8"]

    dicts = []
    for file in files:
        file_path = path + file + "/result.json"
        with open(file_path, 'r') as handle:
            json_data = [json.loads(line) for line in handle]

        dict = {}
        for item in json_data:
            dict[item["time_total_s"]] = item["episode_reward_mean"]
        dicts.append(dict)

    plot(dicts)


def plot(data):

    plt.figure()

    legends = ["central", "concur", "ps"] #, "Central. Critic"]

    i = 0
    for d in data:
        lists = sorted(d.items())
        x, y = zip(*lists)
        #y = savgol_filter(y, 11, 4)

        plt.title("Confusion-3: PPO Centralized")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean episode reward")
        plt.plot(x, y)
        i += 1

    plt.legend(legends, loc=4, fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
