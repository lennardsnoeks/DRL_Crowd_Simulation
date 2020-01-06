import os
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import Visualization


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/urban.xml")
    sim_state = XMLSimulationState(filename).simulation_state
    visualization = Visualization(sim_state)
    visualization.run()


if __name__ == "__main__":
    main()
