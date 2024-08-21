# Automatic Surveillance of Bacteriological Strains within Clinical Settings


This repository contains a series of Python scripts and Jupyter Notebooks implementing the Efficient Peak Information KErnel (E-PIKE) and Standard Peak Removal (SPR) method for MALDI-TOF MS data processing. Specifically, the E-PIKE and SPR method are designed to highlight non-common peak differences between bacteriological isolates belonging to different strains of the same species.

## Paper link

Paper is not yet publicly available.

## Database

Database is not yet publicly available. Stay tuned for more info!

## Overview

The project includes the following key scripts:

1. **E-PIKE (e_pike.py):** Implements the E-PIKE in a redy-to-use Python class.
2. **Scale Normalization (normalization.py):** An adaptation of the Scale Normalization step introduced by C. Weis et al in the [original PIKE implementation](https://github.com/BorgwardtLab/maldi_PIKE).
3. **Kernelization Pipeline pipeline.py):** A ready-to-use class implementing all three pipelines considered in our paper, including the SPR implementation.

## Installation

To set up the project using Conda, follow these steps:

```bash
# Clone the repository
git clone https://github.com/rafardzp/Strain-Surveillance.git
cd Strain Surveillance

# Create a Conda environment from the .yml file
conda env create -f environment.yml

# Activate the Conda environment
conda activate your-env-name
```

## Usage

The Jupyter Notebook `example_usage.ipynb` provides a short example of how to use the `pipeline.py` script to obtain kernelized MALDI-TOF MS samples for each one of the possible pipelines.


## Contributing

Contributions to this project are welcome. Please follow the standard fork-and-pull request workflow on GitHub. If you have any suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Please see the `LICENSE` file for more details.

## Contact

For any queries or further assistance, please reach out to [Msc. Rafael Rodríguez](mailto:rrpalomo@tsc.uc3m.es).
