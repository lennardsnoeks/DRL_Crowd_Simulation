import glob
import json
import matplotlib.pyplot as plt
from pandas import np
from scipy.signal import savgol_filter


def main():
    #plot_reward_progress()

    plot_comparison()


def plot_comparison():
    path = "/home/lennard/ray_results/hallway/"

    types = ["ppo_0", "ppo_8", "ddpg_0"]

    schemes = ["central_", "conc_", "ps_", "centralq_"]

    avg_times_1 = []
    avg_times_8 = []
    avg_times_ddpg = []

    for type in types:
        for scheme in schemes:
            avg_array = []
            for i in range(1, 6):
                if scheme == "central_":
                    avg_array.append(0)
                    break
                if type == "ddpg_0" and scheme == "ps_" and i > 1:
                    break
                if type == "ppo_0" and scheme == "ps_" and i > 1:
                    break
                if type == "ddpg_0" and scheme == "conc_":
                    avg_array.append(0)
                    break
                if type == "ppo_0" and scheme == "conc_":
                    avg_array.append(0)
                    break
                if type == "ddpg_0" and scheme == "centralq_":
                    avg_array.append(0)
                    break
                if type == "ppo_0" and scheme == "centralq_":
                    avg_array.append(0)
                    break
                """if type == "ppo_0" and scheme == "central_" and i > 1:
                    break
                if type == "ddpg_0" and scheme == "centralq_":
                    avg_array.append(0)
                    break
                if type == "ddpg_0" and scheme == "central_":
                    avg_array.append(2059.2)
                    break"""
                file_path = path + scheme + type + "_" + str(i) + "/result.json"
                with open(file_path, 'r') as handle:
                    json_data = [json.loads(line) for line in handle]

                time = []
                for item in json_data:
                    time.append(item["time_total_s"])

                avg_array.append(time[-1])

            avg = sum(avg_array) / len(avg_array)

            if type == "ppo_0":
                avg_times_1.append(avg)
            elif type == "ppo_8":
                avg_times_8.append(avg)
            else:
                avg_times_ddpg.append(avg)

    # set width of bar
    barWidth = 0.25

    print(avg_times_ddpg, avg_times_1, avg_times_8)

    # Set position of bar on X axis
    r1 = np.arange(len(avg_times_ddpg))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth * 2 for x in r1]

    # Make the plot
    plt.bar(r1, avg_times_ddpg, color='#f59642', width=barWidth, edgecolor='white', label='DDPG - 1 worker')
    plt.bar(r2, avg_times_1, color='#4266f5', width=barWidth, edgecolor='white', label='PPO - 1 worker')
    plt.bar(r3, avg_times_8, color='#42bcf5', width=barWidth, edgecolor='white', label='PPO - 8 workers')

    # Add xticks on the middle of the group bars
    #plt.xlabel('3-confusion', fontweight='bold')
    plt.title("2-way confusion", fontweight="bold")
    plt.xticks([r + barWidth for r in range(len(avg_times_1))], ['Centralized', 'Concurrent', 'PS', 'Central critic'])
    plt.ylabel('Time (s)', fontweight='bold')

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

    path = "/home/lennard/ray_results/confusion/"
    files = ["central_ppo_8_2", "conc_ppo_8_2", "ps_ppo_8_3", "centralq_ppo_8_2"]

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

    legends = ["Central", "Concur", "PS", "Central Critic"]

    i = 0
    for d in data:
        lists = sorted(d.items())
        x, y = zip(*lists)
        #y = savgol_filter(y, 11, 4)

        plt.title("2-way confusion: PPO 8 workers", fontweight="bold")
        plt.xlabel("Time (s)", fontweight="bold")
        plt.ylabel("Mean episode reward", fontweight="bold")
        plt.plot(x, y)
        i += 1

    plt.legend(legends, loc=4, fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
