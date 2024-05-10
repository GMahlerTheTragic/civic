[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=GMahlerTheTragic_civic&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=GMahlerTheTragic_civic)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=GMahlerTheTragic_civic&metric=coverage)](https://sonarcloud.io/summary/new_code?id=GMahlerTheTragic_civic)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=GMahlerTheTragic_civic&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=GMahlerTheTragic_civic)
# civic

This repository contains PyTorch code to fine-tune and pretrain various language models (BERT- & RoBERTa-based) for a classification task from the medical domain.
Civic Evidence is the task of labeling medical paper abstracts from PubMed with up to 5 levels of clinical evidence.
The Civic evidence model is explained here: https://civic.readthedocs.io/en/latest/curating/evidence.html


## Table of Contents
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

Here is a brief overview of the project structure:

- **civic/**: Contains the core Python package with PyTorch datasets, training, evaluation code, and models.
- **data/**: Used to store training, validation, and test data. Data processing scripts will save processed data here.
- **models/**: A location to save trained models and snapshots.
- **notebooks/**: Jupyter Notebooks for experimentation and analysis.
- **scripts/**: Scripts for data download, processing, training, and evaluation to be run on a remote server with GPUs.
- **test/**: Contains unit tests to ensure code quality and reliability.

## Setup and Installation

To set up this project, follow these steps:

## Install dependencies

```bash
cd civic
pip install -r requirements.txt
```

## Usage

The following are common usage scenarios for this project:

- **Data Preparation**: Run scripts `download_civic_evidence.py` and `process_civic_evidence_data.py`

- **Model Training**: Run `launch_grid_search.sh` (requires at least two Nvidia A100 for reasonable performance)

- **Model Evaluation**: Run `launch_evaluation.sh` (requires at least two Nvidia A100 for reasonable performance)

- **Analysis and Experimentation**: Use Jupyter Notebooks from `notebooks/` to conduct additional analysis or visualization.

- **Unit Testing**: Ensure code reliability by running tests from the `test/` folder with `pytest`.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes and push to your branch.
4. Open a Pull Request and describe the changes you've made.

## Contact

If you have any questions or need support, please create an issue in the GitHub repository.





