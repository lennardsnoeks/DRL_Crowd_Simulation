import json
import matplotlib.pyplot as plt


def main():
    files = ["central_ok_reset_1_finish",
             "concurrent_good_all_finish",
             "ps_good_all_finish"]
    path = "/home/lennard/ray_results/training_case_1/"

    dicts = []
    for file in files:
        file_path = path + file + "/result.json"
        with open(file_path, 'r') as handle:
            json_data = [json.loads(line) for line in handle]

        dict = {}
        for item in json_data:
            dict[item["timesteps_total"]] = item["episode_reward_mean"]
        dicts.append(dict)

    plot(dicts)


def plot(data):

    plt.figure()

    legends = ["Centralized", "Concurrent", "PS"]

    i = 0
    for d in data:
        lists = sorted(d.items())
        x, y = zip(*lists)

        plt.title("Test case 1: 2-way confusion")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean episode reward")
        plt.plot(x, y)

        i += 1

    plt.legend(legends, loc=4, fontsize="small")
    plt.show()


if __name__ == "__main__":
    main()
