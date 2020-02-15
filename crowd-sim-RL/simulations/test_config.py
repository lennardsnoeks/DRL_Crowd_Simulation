import os

from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import VisualizationLive


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles2.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    visualize(sim_state)


def visualize(sim_state):
    visualization = VisualizationLive(sim_state)
    visualization.run()


if __name__ == "__main__":
    main()
