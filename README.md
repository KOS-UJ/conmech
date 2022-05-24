# conmech

conmech is a simulating package written in Python that allows to numerically solve contact mechanics problems. 

### Description

Contact mechanics describes behaviour of the physical body in contact with the obstacle. Majority of such problems cannot be solved analitically and require numerical procedure. This package implements the Finite Element Method for 2D and 3D bodies and works with static, quasistatic and dynamic problems. It can simulate various physical phenomena, such as friction and obstacle resistance. Additional variables besides displacement, such as the temperature of the body, can be calculated. The project is almost entirely self contained, requires basic Python libraries, [pygmsh](https://github.com/meshpro/pygmsh) along with [Gmsh](https://gmsh.info/) for mesh construction and [Scipy](https://scipy.org/) for solving reformulated problem. The code is modular and can be further extended to include new physical models.

### Sample results

| 2D with temperature | 3D |
:-------------------------:|:-------------------------:
<img src="/samples/circle_roll_temperature.gif" width="100%" /> |  <img src="samples/ball_roll_3d.gif" width="100%" />


### Installation

Install Gmsh used for mesh construction and dependencies from "requirements.txt"

    apt-get install python3-gmsh
    pip install -r requirements.txt

### Usage

To run sample simulations, start any file from examples folder

    PYTHONPATH=. python examples/examples_2d.py


# deep_conmech

deep_conmech uses Graph Neural Networks to learn dynamic contact mechanics simulations based on energy function implemented in conmech. It is implemented in [PyTorch](https://pytorch.org/) using [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) library.

### Sample results

<img src="samples/graph_circle_slope.gif" width="100%" /> | <img src="samples/graph_circle_left.gif" width="100%" />
:-------------------------:|:-------------------------:

### Installation

Follow the instructions to install conmech, and then install additional dependencies by

    pip install -r requirements-deep.txt

### Usage

To generate training and validation sets, start the training process and periodically save learned model parameters, run

    PYTHONPATH=. python deep_conmech/run_model.py --mode=train

To generate sample trajectories using the most current saved model parameters, run

    PYTHONPATH=. python deep_conmech/run_model.py --mode=plot
 


