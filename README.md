# conmech

**conmech** is a Python package designed for the numerical simulation of contact mechanics problems. 
It facilitates the analysis of interactions between physical bodies and obstacles, employing the Finite Element Method (FEM) for both 2D and 3D models. The package supports static, quasistatic, and dynamic analyses, enabling the simulation of phenomena such as friction and obstacle resistance. Additionally, conmech can compute variables like body temperature, providing a comprehensive tool for contact mechanics simulations. 


## Features

- **Finite Element Analysis**: Implements FEM for accurate modeling of contact mechanics in 2D and 3D structures.
- **Versatile Problem Solving**: Capable of handling static, quasistatic, and dynamic scenarios.
- **Friction and Resistance Simulation**: Models complex interactions including frictional forces and obstacle resistance.
- **Multi-Effects Analysis**: Computes temperature, electric etc. distributions within the analyzed bodies.

## Installation

To install conmech, clone the repository and install the required dependencies:

```bash
git clone https://github.com/KOS-UJ/conmech.git
cd conmech
pip install -r requirements.txt
```

## Usage

After installation, you can utilize conmech in your Python scripts as follows:

```python
import conmech

# Example usage
# Initialize your model, define materials, apply boundary conditions, and solve
```

For detailed examples and tutorials, please refer to the [examples directory](https://github.com/KOS-UJ/conmech/tree/master/examples).

## Contributing

Contributions to conmech are welcome and appreciated. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Implement your changes.
4. Submit a pull request detailing your modifications.

Please ensure adherence to the project's coding standards and include relevant tests for new features.

## License

This project is licensed under the [GPL-3.0 License](https://github.com/KOS-UJ/conmech/blob/master/LICENSE).

## Acknowledgements

This project has received funding from 
 - the European Union’s Horizon 2020 Research and Innovation Programme under the Marie Sklodowska-Curie Grant Agreement No 823731
CONMECH,
 -  National Science Center, Poland, under project OPUS no. 2021/41/B/ST1/01636,
 - the program Excellence Initiative - Research University at the Jagiellonian University in Kraków. 

For any questions or support, please open an issue on GitHub or contact the maintainers directly. 