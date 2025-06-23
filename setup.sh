#!/bin/bash
if [ ! -f "Miniforge3.sh" ]; then
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
fi
bash Miniforge3.sh -b -p ./conda
source "./conda/etc/profile.d/conda.sh"
if [ ! -d "./main" ]; then
    conda create -p ./main polars matplotlib seaborn tqdm pdbfixer rdkit openbabel vina gitpython uv numpy questionary pyyaml conda apptainer pymol-open-source -y
    conda run -p ./main uv venv
    conda run -p ./main uv run uv pip install pykvfinder boltz pyarrow
fi
if [ ! -d "diffdock" ]; then
    wget 
    conda env create -f DiffDock/environment.yml
fi
conda activate ./main
python main.py