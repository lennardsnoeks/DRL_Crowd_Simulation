import os

from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import VisualizationLive


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_squeeze_2.xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    visualize(sim_state)


def visualize(sim_state):
    visualization = VisualizationLive(sim_state)
    visualization.run()


if __name__ == "__main__":
    main()
