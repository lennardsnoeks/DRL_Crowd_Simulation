import os

from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_training import VisualizationLive


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/5-crossway_2_groups/group.xml")
    seed = 23
    sim_state = XMLSimulationState(filename, seed).simulation_state

    visualize(sim_state)


def visualize(sim_state):
    zoom_factor = 10

    visualization = VisualizationLive(sim_state, zoom_factor)
    visualization.run()


if __name__ == "__main__":
    main()
