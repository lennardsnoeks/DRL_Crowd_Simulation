import json
import matplotlib.pyplot as plt


def main():
    files = ["ddpg_all"]
    path = "/home/lennard/ray_results/ddpg_speedup/"

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

    #legends = ["Centralized", "Concurrent", "PS"]
    legends = ["a"]

    i = 0
    for d in data:
        lists = sorted(d.items())
        x, y = zip(*lists)

        plt.title("Hallway-4: DDPG no Curriculum Learning")
        plt.xlabel("Time (s)")
        plt.ylabel("Mean episode reward")
        plt.plot(x, y)

        i += 1

    plt.legend(legends, loc=4, fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
