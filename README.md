# clef-project-template

https://www.joker-project.com/clef-2025/
https://clef2025.clef-initiative.eu/index.php?page=Pages/Labs/JOKER.html

## quickstart

Install the package into your environment.
It is useful to install the package in editable mode so that you can make changes to the code and see the changes reflected in your environment.
Use a virtual environment when possible to avoid conflicts with other packages.

```bash
brew install python3
alias python=python3
touch ~/.bashrc
source ~/.bashrc
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd src
python filename.py
```

## structure

The repository structure is as follows:

```
root/
├── src/              # the task package for the project
├── tests/            # tests for the project
├── notebooks/        # notebooks for the project
├── user/             # user-specific directories
├── scripts/          # scripts for the project
└── docs/             # documentation for the project
```

The `src` directory contains the main code for the project, organized into modules and submodules as needed.
The `tests` directory contains the tests for the project, organized into test modules that correspond to the code modules.
The `notebooks` directory contains Jupyter notebooks that capture exploration of the datasets and modeling.
The `user` directory is a scratch directory where users can commit files without worrying about polluting the main repository.
The `scripts` directory contains scripts that are used in the project, such as utility scripts for working with PACE or GCP.
The `docs` directory contains documentation for the project, including explanations of the code, the data, and the models.
