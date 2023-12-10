# MEPS Bias Mitigation using AiF360

## Overview

This repository serves as a case study for applying bias mitigation methods using the AI Fairness 360 (AiF360) toolkit. The primary focus is on addressing biases in the Medical Expenditure Panel Survey (MEPS) dataset. The toolkit is utilized to detect and mitigate biases, providing insights into the potential impacts of bias mitigation techniques in real-world scenarios.

## Getting Started


To replicate our environment, you can choose from the following options. Regardless of which option,
in order to explore our work:

1. Clone the repository:

    ```
    git clone https://github.com/474benchen/capstone_replication.git
    ```

2. Navigate to the project directory:

    ```
    cd capstone_replication
    ```
### Option 1: Using requirements.txt

1. Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

2. Proceed to the next steps for dataset setup and running the bias mitigation methods.

### Option 2: Reconstructing the Conda Environment

Ensure that you have conda installed before attempting this option. distributions can be found [here](https://www.anaconda.com/download).

1. Create a conda environment from the provided environment.yml file:

    ```
    conda env create -f environment.yml
    ```

2. Activate the conda environment:

    ```
    conda activate capstone-aif360
    ```

3. Proceed to the next steps for dataset setup and running the bias mitigation methods.

## Dataset Setup

Before running our notebook, you need to set up the MEPS dataset. We've provided the data in `data.zip`.
In order to access it, unzip the file into the capstone_replication directory. The created data directory
should contain 2 csv files, each correlating to data from a MEPS panel.

## Running Bias Mitigation

Once the environment is set up and the dataset is prepared, refer to `replication.ipynb` to explore our work.

## Contributors

- [Benjamin Chen](https://www.linkedin.com/in/474benjaminchen/)
- [Jayson Leach](https://www.linkedin.com/in/jayson-leach/)
- [Stephanie Chavez](https://www.linkedin.com/in/stephanie-chavez-000840223/)
- [Natalie Wu](https://www.linkedin.com/in/natalie-wu5/)
- [Sam Horio](https://www.linkedin.com/in/samantha-horio/)

