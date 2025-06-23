import os
import sys
import subprocess
import logging
import argparse
import gzip
import shutil
import re
from pathlib import Path
import questionary
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from vina import Vina
import git
import numpy as np
import pyKVFinder
import tempfile
import yaml
import json
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO ,Structure, Model, Chain
from pymol import cmd
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("virtual_screening.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run(command, working_dir=None, env=None):
    """Runs a command in the shell and logs its output, raises error on failure."""
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            env=env,
            shell=True
        )
        stdout_lines, stderr_lines = [], []
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout, end='')  # Print directly to terminal
            logging.info(f"STDOUT:\n{stdout}")
            stdout_lines = stdout.splitlines()
        if stderr:
            print(stderr, end='', file=sys.stderr)  # Print directly to terminal
            logging.warning(f"STDERR:\n{stderr}")
            stderr_lines = stderr.splitlines()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
        return stdout_lines
    except FileNotFoundError:
        logging.error(f"Command not found: {command[0]}. Is it in your PATH?")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        raise e

def calculate_protein_geometry(protein_pdb_path: Path) -> tuple[list, list]:
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(protein_pdb_path))
    
    # Get all atom coordinates
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.coord)
    
    coords = np.array(coords)
    
    # Calculate center of mass (mean of all atom coordinates)
    center = coords.mean(axis=0)
    
    # Calculate box size from maximum protein dimensions with padding
    max_coords = coords.max(axis=0)
    min_coords = coords.min(axis=0)
    box_size = (max_coords - min_coords) + 5.0  # Add 5Å padding
    
    logging.info(f"Protein center of mass: {center.tolist()}")
    logging.info(f"Calculated box size: {box_size.tolist()}")
    
    return center.tolist(), box_size.tolist()

def manual_split_pdbqt(pdbqt_path: Path, split_dir: Path) -> list[Path]:
    """Manually split a multi-ligand PDBQT file into individual ligand files."""
    split_files = []
    
    with open(pdbqt_path, 'r') as f:
        content = f.read()
    
    # Split by ROOT/ENDROOT pairs or MODEL/ENDMDL pairs
    if 'ROOT' in content:
        # Split by ROOT/ENDROOT
        ligand_blocks = re.split(r'(?=ROOT)', content)
        ligand_blocks = [block for block in ligand_blocks if block.strip()]  # Remove empty blocks
        
        for i, block in enumerate(ligand_blocks):
            if block.strip():
                ligand_file = split_dir / f"ligand{i+1:02d}.pdbqt"
                with open(ligand_file, 'w') as f:
                    f.write(block)
                split_files.append(ligand_file)
    
    elif 'MODEL' in content:
        # Split by MODEL/ENDMDL
        ligand_blocks = re.split(r'MODEL\s+\d+', content)
        ligand_blocks = [block for block in ligand_blocks if block.strip()]  # Remove empty blocks
        
        for i, block in enumerate(ligand_blocks):
            if block.strip():
                ligand_file = split_dir / f"ligand{i+1:02d}.pdbqt"
                with open(ligand_file, 'w') as f:
                    f.write(f"MODEL {i+1}\n{block}")
                split_files.append(ligand_file)
    
    return split_files

def split_pdb_by_chain(pdb_path: Path, chain_id: str, output_path: Path) -> Path:
    """Extracts a specific chain from a PDB file and writes it to output_path using Bio.PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', str(pdb_path))
    io = PDBIO()
    # Find the chain
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                new_structure = Structure.Structure('single_chain')
                new_model = Model.Model(0)
                new_chain = Chain.Chain(chain_id)
                for residue in chain:
                    new_chain.add(residue)
                new_model.add(new_chain)
                new_structure.add(new_model)
                io.set_structure(new_structure)
                io.save(str(output_path))
                return output_path
    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

def create_directories(base_path: Path):
    """Create the necessary directory structure for the results."""
    base_path.mkdir(exist_ok=True)
    (base_path / "protein").mkdir(exist_ok=True)
    (base_path / "ligands").mkdir(exist_ok=True)
    (base_path / "docking_results").mkdir(exist_ok=True)
    (base_path / "plots").mkdir(exist_ok=True)
    (base_path / "final_report").mkdir(exist_ok=True)
    logging.info(f"Created directory structure in {base_path}")


# --- Workflow Functions ---

def get_user_preferences():
    """Get all user preferences at the start of the program."""
    global PROTEIN_INPUT_METHOD, LIGAND_INPUT_METHOD, DOCKING_METHOD, RESCORING_METHOD, VIEW_INTERACTIONS, pdb_id , smiles , path_str , num_to_extract , sequence , fold_method
    
    # Get protein input method
    PROTEIN_INPUT_METHOD = questionary.select(
        "How do you want to provide the protein?",
        choices=["PDB ID", "Local PDB file", "Sequence", "Skip"]
    ).ask()
    
    if PROTEIN_INPUT_METHOD == "PDB ID":
        pdb_id = questionary.text("Enter the 4-character PDB ID:").ask().upper()
    if PROTEIN_INPUT_METHOD == "Sequence":
        sequence = questionary.text("Enter the amino acid sequence:").ask().upper()
        fold_method = questionary.select(
            "Which service to use for structure prediction?",
            choices=["ESMFold (Fast, Recommended)", "Boltz (Slow)"]
        ).ask()
        
    # Get ligand input method
    LIGAND_INPUT_METHOD = questionary.select(
        "How do you want to provide the ligand(s)?",
        choices=[
            "SDF/MOL2", 
            "SMILES", 
            "ChEMBL", 
            "Test"
        ]
    ).ask()
    if LIGAND_INPUT_METHOD == "SDF/MOL2":
        path_str = questionary.path("Enter the path to your ligand file:").ask()
    if LIGAND_INPUT_METHOD == "SMILES":
        smiles = questionary.text("Enter the SMILES string:").ask()
    if LIGAND_INPUT_METHOD == "ChEMBL":
        num_to_extract = min(int(questionary.text(
            f"How many ligands to randomly extract from chembl? (max 2496335)", default="100"
        ).ask()), 2496335)    
    # Get docking method
    DOCKING_METHOD = questionary.select(
        "Which docking engine do you want to use?",
        choices=["Autodock-Vina", "DiffDock (Requires separate env)", "Boltz (Experimental)"]
    ).ask()
    
    # Get rescoring method
    RESCORING_METHOD = questionary.select(
        "Which tool to use for rescoring?",
        choices=["Vina", "Gnina (requires download)"]
    ).ask()
    
    # Get view interactions preference
    VIEW_INTERACTIONS = questionary.select(
        "Would you like to view 2D protein-ligand interactions?",
        choices=["Yes", "No"]
    ).ask()
    
    logging.info("User preferences collected:")
    logging.info(f"  Protein input method: {PROTEIN_INPUT_METHOD}")
    logging.info(f"  Ligand input method: {LIGAND_INPUT_METHOD}")
    logging.info(f"  Docking method: {DOCKING_METHOD}")
    logging.info(f"  Rescoring method: {RESCORING_METHOD}")
    logging.info(f"  View interactions: {VIEW_INTERACTIONS}")

def create_complex_file(protein_fixed_path: Path, pose_path: Path, ligand_id: str, docking_tool: str, output_dir: Path) -> Path:
    """Create a complex file combining protein and ligand structures. """
    complex_path = output_dir / f"{ligand_id}_{docking_tool}_complex.pdb"
    
    try:
        # # Combine protein and ligand into single PDB file
        # with open(complex_path, 'w') as complex_file:
        #     # Copy protein structure
        #     with open(protein_fixed_path, 'r') as protein_file:
        #         complex_file.write(protein_file.read())
            
        #     # Add ligand structure (convert SDF to PDB if needed)
        #     if pose_path.suffix == ".sdf":
        #         ligand_pdb = pose_path.with_suffix(".pdb")
        #         run(f"obabel {pose_path} -O {ligand_pdb}")
        #         with open(ligand_pdb, 'r') as ligand_file:
        #             complex_file.write(ligand_file.read())
        #         ligand_pdb.unlink()  # Clean up temporary file
        #     else:
        #         with open(pose_path, 'r') as ligand_file:
        #             complex_file.write(ligand_file.read())
        
        # logging.info(f"Complex file created: {complex_path}")
        # return complex_path
        if pose_path.suffix == ".sdf":
            ligand_pdb_2 = pose_path.with_suffix(".pdb")
            run(f"obabel {pose_path} -O {ligand_pdb_2}")
        elif pose_path.suffix == ".pdbqt":
            ligand_pdb_2 = pose_path.with_suffix(".pdb")
            run(f"obabel {pose_path} -O {ligand_pdb_2}")
        else:
            ligand_pdb_2 = pose_path
        cmd.load(protein_fixed_path,"prot")
        cmd.load(ligand_pdb_2,"lig")
        cmd.create("comp","prot or lig")
        cmd.save(complex_path,"comp")
        logging.info(f"Complex file created: {complex_path}")
        return complex_path
    except Exception as e:
        logging.error(f"Failed to create complex file for {ligand_id}: {e}")
        raise

def get_protein_input(protein_dir: Path) -> Path:
    """Get protein input from the user based on global preference."""
    method = PROTEIN_INPUT_METHOD

    if method == "PDB ID":
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_path = protein_dir / f"protein.pdb"
        logging.info(f"Downloading PDB from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(pdb_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Saved to {pdb_path}")
        return pdb_path

    elif method == "Local PDB file":
        pdb_path_str = questionary.path("Enter the path to your PDB file:").ask()
        pdb_path = Path(pdb_path_str)
        if not pdb_path.exists():
            logging.error(f"File not found: {pdb_path}")
            sys.exit(1)
        shutil.copy2(pdb_path, protein_dir / "protein.pdb")
        return pdb_path

    elif method == "Sequence":
        
        pdb_path = protein_dir / "protein.pdb"
        
        if fold_method.startswith("ESMFold"):
            url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            logging.info("Predicting structure using ESMFold API...")
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            with requests.post(url, data=sequence, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(pdb_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            logging.info(f"ESMFold prediction saved to {pdb_path}")
            return pdb_path

        else:  # Boltz
            logging.info("Predicting structure using Boltz...")
            try:
                # Use the Boltz CLI for structure prediction, as recommended by the official documentation.
                # Write the sequence to a temporary FASTA file as required by Boltz.
                input_fasta = protein_dir / "input.fasta"
                with open(input_fasta, "w") as f:
                    f.write(f">A|protein\n{sequence}\n")
                run(f"uv run boltz predict {input_fasta} --out_dir {protein_dir} --output_format pdb --use_msa_server --cache boltz_cache --accelerator cpu")
                logging.info(f"Boltz prediction saved to {pdb_path}")
                shutil.copy(protein_dir / "predictions/input/input_model_0.pdb", protein_dir / "protein.pdb")
            except Exception as e:
                logging.error(f"Boltz prediction failed: {e}")
                sys.exit(1)
            return pdb_path
    elif method == "Skip":
        logging.info("Skipping protein input...")
        pdb_path=protein_dir / "protein.pdb"
        return pdb_path
    return None

def prepare_protein(pdb_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Cleans a PDB file and converts to PDBQT."""
    protein_name = pdb_path.stem
    fixed_pdb_path = output_dir / f"{protein_name}_fixed.pdb"
    protein_pdbqt_path = output_dir / f"{protein_name}_protein.pdbqt"

    logging.info("--- Preparing Protein ---")
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(False)
    #fixer.addMissingHydrogens()
    with open(fixed_pdb_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    logging.info(f"Protein fixed and saved to {fixed_pdb_path}")
    
    logging.info("Converting to PDBQT format...")
    run(f"obabel {fixed_pdb_path} -O {protein_pdbqt_path} -xr -h")
    logging.info(f"Protein PDBQT saved to {protein_pdbqt_path}")
    
    return fixed_pdb_path, protein_pdbqt_path

def get_ligand_input(ligand_dir: Path) -> list[Path]:
    """Get ligand input from the user based on global preference."""
    global smiles  
    method = LIGAND_INPUT_METHOD

    ligand_files = []
    
    if method == "SDF/MOL2":
        ligand_path = Path(path_str)
        if not ligand_path.exists():
            logging.error(f"File not found: {ligand_path}")
            sys.exit(1)
        
        # Convert to a standard SDF format if it's MOL2
        if ligand_path.suffix.lower() == ".mol2":
            output_sdf = ligand_dir / f"{ligand_path.stem}.sdf"
            run(f"obabel -imol2 {ligand_path} -osdf -O {output_sdf}")
            ligand_files.append(output_sdf)
        else:
            ligand_files.append(ligand_path)

    elif method == "SMILES":
        name = "ligand"
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logging.error("Invalid SMILES string provided.")
            sys.exit(1)
        
        output_sdf = ligand_dir / f"{name}.sdf"
        with Chem.SDWriter(str(output_sdf)) as w:
            w.write(mol)
        ligand_files.append(output_sdf)

    elif method.startswith("ChEMBL"):
        url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.sdf.gz"
        gz_path = ligand_dir / "chembl_35.sdf.gz"
        sdf_path = ligand_dir / "chembl_35.sdf"
        
        if not sdf_path.exists():
            logging.info(f"Downloading ChEMBL database from {url} (this is large)...")
            run(f"wget -O {gz_path} {url}")
            
            logging.info("Decompressing ChEMBL database...")
            with gzip.open(gz_path, 'rb') as f_in, open(sdf_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink() # Clean up
        
        extracted_sdf_path = ligand_dir / f"chembl_extracted_{num_to_extract}.sdf"
        logging.info(f"Extracting {num_to_extract} ligands...")
        
        # Extract ligands
        supplier = Chem.ForwardSDMolSupplier(str(sdf_path))
        with Chem.SDWriter(str(extracted_sdf_path)) as writer:
            count = 0
            for mol in supplier:
                if mol is not None:
                    writer.write(mol)
                    count += 1
                    if count >= num_to_extract:
                        break
        writer.close()
        ligand_files.append(extracted_sdf_path)

    elif method.startswith("Test"):
        test_smiles = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
            "paracetamol": "CC(=O)NC1=CC=C(O)C=C1"
        }
        test_sdf_path = ligand_dir / "test_set.sdf"
        with Chem.SDWriter(str(test_sdf_path)) as writer:
            for name, smiles in test_smiles.items():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol.SetProp("_Name", name)
                    writer.write(mol)
                else:
                    logging.warning(f"Invalid SMILES for {name}: {smiles}")
        ligand_files.append(test_sdf_path)
    return ligand_files

def prepare_ligands(sdf_files: list[Path], output_dir: Path) -> list[Path]:
    """Converts ligand SDF files to PDBQT format and prepares them for docking."""
    logging.info("--- Preparing Ligands ---")
    prepared_ligands_dir = output_dir / "prepared_ligands"
    prepared_ligands_dir.mkdir(exist_ok=True)
    
    pdbqt_files = []
    
    all_mols = []
    for sdf_file in sdf_files:
        supplier = Chem.SDMolSupplier(str(sdf_file))
        for mol in supplier:
            if mol is not None:
                all_mols.append(mol)

    # First prepare all ligands (3D coordinates, optimization)
    prepared_sdf = prepared_ligands_dir / "prepared_ligands.sdf"
    with Chem.SDWriter(str(prepared_sdf)) as writer:
        success = 0
        fail = 0
        max_fail = 1000  # Stop after 1000 failures to avoid infinite loops
        for i, mol in enumerate(all_mols):
            try:
                mol = Chem.AddHs(mol)
                # Try multiple times with different random seeds if embedding fails
                res = -1
                for seed in [42, 123, 456, 789, 999]:  # Try multiple random seeds
                    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
                    if res == 0:
                        break
                
                if res != 0:
                    logging.warning(f"Failed to embed molecule {i+1}: Could not generate 3D coordinates")
                    fail += 1
                    if fail >= max_fail:
                        logging.warning(f"Reached {max_fail} failed ligand preparations, stopping early.")
                        break
                    continue
                
                AllChem.MMFFOptimizeMolecule(mol)
                # Ensure molecule has a name for tracking
                if not mol.HasProp('_Name'):
                    mol.SetProp('_Name', f'ligand_{success+1}')
                writer.write(mol)
                success += 1
            except Exception as e:
                fail += 1
                logging.warning(f"Failed to prepare ligand {i+1}: {e}")
                if fail >= max_fail:
                    logging.warning(f"Reached {max_fail} failed ligand preparations, stopping early.")
                    break
                continue
        logging.info(f"Prepared {success} ligands successfully, {fail} failed.")

    # Then convert to PDBQT format
    logging.info("Converting prepared ligands to PDBQT format...")
    pdbqt_path = prepared_ligands_dir / "prepared_ligands.pdbqt"
    run(f"obabel {prepared_sdf} -O {pdbqt_path}")

    # Check if the PDBQT file contains multiple ligands by counting molecules
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()
        
        # Count MODEL tags for multi-model format
        model_count = sum(1 for line in lines if line.strip().startswith('MODEL'))
        
        # Count REMARK Name lines to determine number of unique ligands
        remark_names = []
        for line in lines:
            if line.strip().startswith('REMARK  Name ='):
                name = line.strip().split('=')[1].strip()
                if name not in remark_names:
                    remark_names.append(name)
        
        # If we have MODEL tags, use that count
        # Otherwise, the number of unique REMARK Names indicates ligand count
        if model_count > 0:
            ligand_count = model_count
        else:
            ligand_count = len(remark_names) if remark_names else 1  # Default to 1 if no names found

    if ligand_count > 1:
        # Split multi-model PDBQT into individual ligand PDBQT files
        logging.info(f"Splitting PDBQT with {ligand_count} ligands into individual files...")
        split_dir = prepared_ligands_dir / "split_pdbqts"
        split_dir.mkdir(exist_ok=True)
        # Add a suffix based on the ligand input method
        suffix = f"_{LIGAND_INPUT_METHOD.lower()}" if 'LIGAND_INPUT_METHOD' in globals() else ""
        split_prefix = split_dir / f"ligand{suffix}"
        
        try:
            run(f"vina_split --input {pdbqt_path} --ligand {split_prefix}")
            split_files = list(split_dir.glob(f"ligand{suffix}*.pdbqt"))
            if not split_files:
                # If vina_split didn't work, manually split the file
                logging.info("vina_split produced no files, attempting manual split...")
                split_files = manual_split_pdbqt(pdbqt_path, split_dir)
        except subprocess.CalledProcessError as e:
            # If vina_split fails, use manual splitting as fallback
            logging.warning(f"vina_split failed with error: {e}")
            logging.info("Attempting manual split as fallback...")
            split_files = manual_split_pdbqt(pdbqt_path, split_dir)
        
        if split_files:
            logging.info(f"Prepared {len(split_files)} individual ligand PDBQT files.")
            return split_files
        else:
            # If both methods fail, just return the original file's path as a single-item list
            logging.warning("Both vina_split and manual split failed, using original PDBQT file")
            return [pdbqt_path]
    else:
        logging.info("Only one ligand found, skipping split.")
        return [pdbqt_path]

def run_docking(
    protein_pdb_path: Path, 
    protein_pdbqt_path: Path, 
    ligand_pdbqts: list[Path],
    ligand_sdfs: list[Path],
    output_dir: Path
) -> pl.DataFrame:
    """Run docking using the globally selected method."""
    method = DOCKING_METHOD

    docking_results_dir = output_dir / "docking_results"
    
    if method == "Autodock-Vina":
        logging.info("--- Running Autodock-Vina Docking ---")
        
        # Calculate protein center of mass and box size from protein coordinates
        logging.info("Calculating protein center of mass and box size...")
        center, box_size = calculate_protein_geometry(protein_pdb_path)
        v = Vina(sf_name='vina')
        v.set_receptor(str(protein_pdbqt_path))

        results = []
        for ligand_pdbqt in ligand_pdbqts:
            try:
                v.set_ligand_from_file(str(ligand_pdbqt))
                v.compute_vina_maps(center=center, box_size=box_size)
                v.dock(exhaustiveness=32, n_poses=10)
                
                pose_path = docking_results_dir / f"{ligand_pdbqt.stem}_vina_docked.pdbqt"
                top_pose_path = docking_results_dir / f"{ligand_pdbqt.stem}_top_pose.pdbqt"
                v.write_poses(str(pose_path), overwrite=True, n_poses=10)
                v.write_poses(str(top_pose_path), overwrite=True, n_poses=1)
                
                # The first energy is the best score
                energies = v.energies(n_poses=1)
                score = energies[0][0]
                
                results.append({
                    "ligand_id": ligand_pdbqt.stem,
                    "docking_tool": "Autodock-Vina",
                    "pose_file": str(pose_path),
                    "docking_score": score # Lower is better
                })
            except Exception as e:
                logging.error(f"Vina docking failed for {ligand_pdbqt.stem}: {e}")
        
        # Create complex files for each docked pose
        logging.info("Creating complex files for all docked poses...")
        logging.info(results)
        for result in results:
            try:
                pose_path = Path(result['pose_file'])
                ligand_id = result['ligand_id']
                docking_tool = result['docking_tool']
                top_pose_path = docking_results_dir / f"{ligand_id}_top_pose.pdbqt"
                complex_path = create_complex_file(
                    protein_pdb_path, top_pose_path, ligand_id, docking_tool, docking_results_dir
                )
                
                # Store complex file path in result for later use
                result['complex_file'] = str(complex_path)
                
            except Exception as e:
                logging.error(f"Failed to create complex file for {ligand_id}: {e}")
        
        return pl.DataFrame(results)

    elif method.startswith("DiffDock"):
        logging.info("Running DiffDock inference...")
        diffdock_output_dir = docking_results_dir / "diffdock_output"
        diffdock_output_dir.mkdir(exist_ok=True)
        
        # Change to DiffDock directory and run inference
        original_cwd = Path.cwd()
        try:
            os.chdir("DiffDock")
            
            # Build command to run inference using the specific environment's python
            command_parts = [
                "conda run", "-p", "../conda/envs/diffdock", "python", "inference.py",
                "--protein_path", str(Path("../") / protein_pdb_path),
                "--out_dir", str(Path("../") / diffdock_output_dir),
                "--inference_steps", "20",
                "--samples_per_complex", "5", "--config", "default_inference_args.yaml"
            ]
            # DiffDock needs individual ligand files
            for sdf_path in ligand_sdfs:
                 command_parts.extend(["--ligand", str(Path("../") / sdf_path)])
            
            # Convert list to string for shell execution
            command = " ".join(command_parts)
            run([command])
        finally:
            os.chdir(original_cwd)

        # Parse DiffDock results
        results = []
        for complex_dir in diffdock_output_dir.iterdir():
            if complex_dir.is_dir() and complex_dir.name.startswith("complex_"):
                top_pose_file = next(complex_dir.glob("rank1_*.sdf"), None)
                if top_pose_file:
                    ligand_name = complex_dir.name.split('_')[-1]
                    match = re.search(r'confidence-([-0-9.]+)\.sdf', top_pose_file.name)
                    score = float(match.group(1)) if match else None
                    results.append({
                        "ligand_id": ligand_name,
                        "docking_tool": "DiffDock",
                        "pose_file": str(top_pose_file),
                        "docking_score": score, # Higher is better
                    })
        
        # Create complex files for each docked pose
        logging.info("Creating complex files for all docked poses...")
        
        for result in results:
            try:
                pose_path = Path(result['pose_file'])
                ligand_id = result['ligand_id']
                docking_tool = result['docking_tool']
                
                complex_path = create_complex_file(
                    protein_pdb_path, pose_path, ligand_id, docking_tool, docking_results_dir
                )
                
                # Store complex file path in result for later use
                result['complex_file'] = str(complex_path)
                
            except Exception as e:
                logging.error(f"Failed to create complex file for {ligand_id}: {e}")
        
        return pl.DataFrame(results)

    elif method.startswith("Boltz"):
        logging.info("--- Running Boltz Docking & Affinity Prediction ---")
        logging.warning("Boltz integration is experimental and requires an API key.")
        
        # Extract protein sequence from PDB file using BioPython
        protein_pdb_path = Path("screening_results/protein/protein.pdb")
        protein_sequence = ""
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", protein_pdb_path)
            
            # Extract sequence from the first chain
            ppb = PDB.PPBuilder()
            for pp in ppb.build_peptides(structure):
                protein_sequence += str(pp.get_sequence())
                
        except Exception as e:
            logging.error(f"Failed to extract protein sequence: {e}")
            sys.exit(1)
        
        # Extract ligand SMILES from all provided SDF files using RDKit
        ligand_smiles_list = []
        try:
            for sdf_path in ligand_sdfs:
                suppl = Chem.SDMolSupplier(str(sdf_path))
                for mol in suppl:
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        ligand_smiles_list.append(smiles)
        except Exception as e:
            logging.error(f"Failed to extract ligand SMILES: {e}")
            sys.exit(1)
        
        results = []
        for i, smiles in enumerate(ligand_smiles_list):
            try:
                # Create input YAML for Boltz
                input_yaml = {
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": protein_sequence
                            }
                        },
                        {
                            "ligand": {
                                "id": "B",
                                "smiles": smiles
                            }
                        }
                    ],
                    "properties": [
                        {
                            "affinity": {
                                "binder": "A"
                            }
                        }
                    ]
                }
                
                # Save YAML to temporary file
                yaml_path = docking_results_dir / f"ligand_{i}_boltz_input.yaml"
                with open(yaml_path, "w") as f:
                    yaml.dump(input_yaml, f)

                # Run Boltz prediction using the CLI as documented
                run(f"uv run boltz predict {yaml_path} --use_msa_server --cache boltz_cache --output_format pdb --accelerator cpu")
                
                # Parse results from Boltz output directory structure
                # Based on Boltz documentation, predictions are saved in predictions/[input_file]/ directory
                input_name = yaml_path.stem
                predictions_dir = Path("predictions") / input_name
                
                # Get the best prediction (model_0 is highest confidence)
                predicted_pdb = predictions_dir / f"{input_name}_model_0.pdb"
                confidence_json = predictions_dir / f"confidence_{input_name}_model_0.json"
                
                if not predicted_pdb.exists():
                    raise FileNotFoundError(f"Boltz prediction file not found: {predicted_pdb}")
                
                # Parse confidence scores
                with open(confidence_json) as f:
                    confidence_data = json.load(f)
                
                # Copy the predicted structure to our results directory
                pose_path = docking_results_dir / f"ligand_{i}_boltz_docked.pdb"
                shutil.copy(predicted_pdb, pose_path)

                results.append({
                    "ligand_id": f"ligand_{i}",
                    "docking_tool": "Boltz",
                    "pose_file": str(pose_path),
                    "docking_score": confidence_data["confidence_score"]  # Higher is better (0-1 range)
                })
                
                # Cleanup temporary files
                yaml_path.unlink()
            except Exception as e:
                logging.error(f"Boltz prediction failed for ligand_{i}: {e}")

        # Create complex files for each docked pose
        logging.info("Creating complex files for all docked poses...")
        
        for result in results:
            try:
                pose_path = Path(result['pose_file'])
                ligand_id = result['ligand_id']
                docking_tool = result['docking_tool']
                
                complex_path = create_complex_file(
                    protein_pdb_path, pose_path, ligand_id, docking_tool, docking_results_dir
                )
                
                # Store complex file path in result for later use
                result['complex_file'] = str(complex_path)
                
            except Exception as e:
                logging.error(f"Failed to create complex file for {ligand_id}: {e}")
        
        return pl.DataFrame(results)
    
    else:
        logging.error(f"Unsupported docking method: {method}")
        return pl.DataFrame()
    


def run_rescoring(database_df: pl.DataFrame, protein_pdbqt: Path, output_dir: Path) -> pl.DataFrame:
    """Rescores top poses using the globally selected method."""
    if database_df.is_empty():
        logging.warning("Docking database is empty, skipping rescoring.")
        return database_df

    method = RESCORING_METHOD

    # Select top 10 poses to rescore
    # Need to handle different sorting for scores (Vina=lower is better, DiffDock=higher is better)
    top_poses = []
    for tool in database_df['docking_tool'].unique():
        df_tool = database_df.filter(pl.col('docking_tool') == tool)
        is_descending = tool != 'Autodock-Vina'
        top_poses.append(df_tool.sort('docking_score', descending=is_descending).head(10))
    
    poses_to_rescore = pl.concat(top_poses).unique(subset=['ligand_id', 'docking_tool'])
    logging.info(f"Selected {len(poses_to_rescore)} unique top poses for rescoring.")
    logging.info(f"{top_poses}")
    
    rescoring_results = []
    
    if method == "Vina":
        v = Vina(sf_name='vina')
        v.set_receptor(str(protein_pdbqt))
        
        # Calculate protein center of mass and box size for affinity maps
        protein_pdb_path = protein_pdbqt.parent / "protein_fixed.pdb"
        center, box_size = calculate_protein_geometry(protein_pdb_path)
        
        # Compute affinity maps once for all rescoring
        v.compute_vina_maps(center=center, box_size=box_size)
        
        for row in poses_to_rescore.iter_rows(named=True):
            pose_path = Path(row['pose_file'])
            ligand_pdbqt_for_rescore = pose_path
            is_diffdock = row['docking_tool'] == 'DiffDock'
            is_boltz = row['docking_tool'] == 'Boltz'
            temp_files = []
            # DiffDock: convert SDF to PDBQT
            if is_diffdock and pose_path.suffix == '.sdf':
                converted_pdbqt = pose_path.with_suffix('.rescoring.pdbqt')
                try:
                    run(f"obabel {pose_path} -O {converted_pdbqt}")
                    ligand_pdbqt_for_rescore = converted_pdbqt
                    temp_files.append(converted_pdbqt)
                except Exception as e:
                    logging.error(f"Failed to convert DiffDock SDF to PDBQT for {row['ligand_id']}: {e}")
                    continue
            # Boltz: split complex into protein (A) and ligand (B), convert both to PDBQT
            if is_boltz and pose_path.suffix == '.pdb':
                split_dir = pose_path.parent / f"split_{pose_path.stem}"
                split_dir.mkdir(exist_ok=True)
                protein_pdb = split_dir / f"{pose_path.stem}_proteinA.pdb"
                ligand_pdb = split_dir / f"{pose_path.stem}_ligandB.pdb"
                split_pdb_by_chain(pose_path, 'A', protein_pdb)
                split_pdb_by_chain(pose_path, 'B', ligand_pdb)
                protein_pdbqt = protein_pdb.with_suffix('.pdbqt')
                ligand_pdbqt = ligand_pdb.with_suffix('.pdbqt')
                try:
                    run(f"obabel {protein_pdb} -O {protein_pdbqt}")
                    run(f"obabel {ligand_pdb} -O {ligand_pdbqt}")
                    ligand_pdbqt_for_rescore = ligand_pdbqt
                    temp_files.extend([protein_pdb, ligand_pdb, protein_pdbqt, ligand_pdbqt])
                except Exception as e:
                    logging.error(f"Failed to convert Boltz split PDBs to PDBQT for {row['ligand_id']}: {e}")
                    continue
            try:
                if is_diffdock:
                    v.set_ligand_from_file(str(ligand_pdbqt_for_rescore))
                    score = v.score()[0]
                    rescoring_results.append({
                        "ligand_id": row['ligand_id'],
                        "docking_tool": row['docking_tool'],
                        "vina_score": score
                    })
                elif is_boltz:
                    # For Boltz, use split protein as receptor, ligand as ligand
                    v.set_receptor(str(protein_pdbqt))
                    v.set_ligand_from_file(str(ligand_pdbqt_for_rescore))
                    score = v.score()[0]
                    rescoring_results.append({
                        "ligand_id": row['ligand_id'],
                        "docking_tool": row['docking_tool'],
                        "vina_score": score
                    })
                else:
                    split_dir = ligand_pdbqt_for_rescore.parent / "split_poses"
                    split_dir.mkdir(exist_ok=True)
                    split_base = split_dir / ligand_pdbqt_for_rescore.stem
                    run(f"vina_split --input {ligand_pdbqt_for_rescore} --ligand {split_base}")
                    split_files = list(split_dir.glob(f"{ligand_pdbqt_for_rescore.stem}*.pdbqt"))
                    logging.info(f"Split directory contents: {[f.name for f in split_dir.iterdir()]}")
                    if not split_files:
                        logging.error(f"vina_split did not produce any split files for {ligand_pdbqt_for_rescore}. Skipping this ligand.")
                        continue
                    v.set_ligand_from_file(str(split_files[0]))
                    score = v.score()[0]
                    rescoring_results.append({
                        "ligand_id": row['ligand_id'],
                        "docking_tool": row['docking_tool'],
                        "vina_score": score
                    })
            except Exception as e:
                logging.error(f"Failed to rescore {row['ligand_id']} with Vina: {e}")
                continue
            finally:
                for f in temp_files:
                    try:
                        Path(f).unlink()
                    except Exception:
                        pass

    elif method == "Gnina (requires download)":
        gnina_exe = Path("./gnina")
        if not gnina_exe.exists():
            logging.info("Downloading Gnina executable...")
            url = "https://github.com/gnina/gnina/releases/download/v1.3.1/gnina1.3.1"
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(gnina_exe, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                gnina_exe.chmod(0o755)
                logging.info("Gnina downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download Gnina: {e}")
                return database_df

        # Calculate protein center and box size for gnina
        protein_pdb_path = protein_pdbqt.parent / "protein_fixed.pdb"
        center, box_size = calculate_protein_geometry(protein_pdb_path)
        
        for row in poses_to_rescore.iter_rows(named=True):
            pose_path = Path(row['pose_file'])
            ligand_for_rescore = pose_path
            
            try:
                logging.info(f"Running Gnina rescoring for {row['ligand_id']}...")
                result = run(f"./gnina -r {protein_pdbqt} -l {pose_path} --score_only --cnn default2018")
                
                # Parse gnina output to extract CNN score
                score = None
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "CNNscore" in line or "Affinity" in line:
                        # Try to extract score from different possible formats
                        parts = line.split()
                        for i, part in enumerate(parts):
                            try:
                                if "CNNscore" in part and i + 1 < len(parts):
                                    score = float(parts[i + 1])
                                    break
                                elif "Affinity" in part and i + 1 < len(parts):
                                    score = float(parts[i + 1])
                                    break
                            except ValueError:
                                continue
                        if score is not None:
                            break
                
                if score is not None:
                    rescoring_results.append({
                        "ligand_id": row['ligand_id'],
                        "docking_tool": row['docking_tool'],
                        "gnina_score": score # Higher is better
                    })
                    logging.info(f"Gnina score for {row['ligand_id']}: {score}")
                else:
                    logging.warning(f"Could not extract Gnina score for {row['ligand_id']}")
                    logging.debug(f"Gnina output: {result.stdout}")
                    
            except subprocess.CalledProcessError as e:
                logging.error(f"Gnina rescoring failed for {row['ligand_id']}: {e}")
                logging.error(f"Command: gnina -r {protein_pdbqt} -l {pose_path} --score_only --cnn default2018")
                logging.error(f"Error output: {e.stderr}")
            except Exception as e:
                logging.error(f"Gnina rescoring failed for {row['ligand_id']}: {e}")

    if not rescoring_results:
        return database_df
        
    rescoring_df = pl.DataFrame(rescoring_results)
    return database_df.join(rescoring_df, on=["ligand_id", "docking_tool"], how="left")

def run_plip_interactions(output_dir: Path):
    """If VIEW_INTERACTIONS is 'Yes', download and run PLIP, then open PyMOL session."""
    global VIEW_INTERACTIONS
    if VIEW_INTERACTIONS != 'Yes':
        logging.info("Skipping PLIP interaction analysis as per user preference.")
        return

    plip_simg_url = "https://github.com/pharmai/plip/releases/download/v2.4.0/plip_2.4.0.simg"
    plip_simg_path = Path("plip.simg")  # Always in current directory
    plip_output_dir = output_dir / "plip"
    plip_output_dir.mkdir(exist_ok=True)
    complex_file = output_dir / "docking_results" / "prepared_ligands_Autodock-Vina_complex.pdb"

    if not complex_file.exists():
        logging.warning(f"Complex file for PLIP analysis not found: {complex_file}. Skipping interaction analysis.")
        return

    if not plip_simg_path.exists():
        logging.info(f"Downloading PLIP singularity image from {plip_simg_url}...")
        with requests.get(plip_simg_url, stream=True) as r:
            r.raise_for_status()
            with open(plip_simg_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded PLIP image to {plip_simg_path}")
    else:
        logging.info(f"PLIP image already exists at {plip_simg_path}")

    # Run PLIP using singularity, referencing plip.simg in current directory
    plip_cmd = f"singularity run {plip_simg_path} -f {complex_file} -o {plip_output_dir} -yvxtp"
    try:
        run(plip_cmd)
        logging.info("PLIP analysis completed.")
    except Exception as e:
        logging.error(f"PLIP analysis failed: {e}")
        return

    # Find and open .pse files in PyMOL
    pse_files = list(plip_output_dir.glob("*.pse"))
    if pse_files:
        pse_files_str = " ".join(str(f) for f in pse_files)
        pymol_cmd = ["pymol"] + [str(f) for f in pse_files] + [f"{output_dir}/protein/protein_fixed.pdb"]
        try:
            run(pymol_cmd)
            logging.info(f"Opened PyMOL session(s) for: {pse_files_str}")
        except Exception as e:
            logging.error(f"Failed to open PyMOL session: {e}")
    else:
        logging.warning(f"No .pse files found in {plip_output_dir} to open in PyMOL.")

def analyze_and_save_results(database_df: pl.DataFrame, output_dir: Path):
    """Analyzes the final dataframe and saves plots and reports."""
    if database_df.is_empty():
        logging.info("Database is empty. No analysis to perform.")
        return
        
    logging.info("--- Analyzing Results and Generating Report ---")
    plots_dir = output_dir / "plots"
    report_dir = output_dir / "final_report"
    
    # Debug: Print database info
    logging.info(f"Database shape: {database_df.shape}")
    logging.info(f"Database columns: {database_df.columns}")
    logging.info(f"First few rows:")
    logging.info(database_df.head().to_pandas().to_string())
    
    # Save the full database in multiple formats
    csv_path = report_dir / "full_screening_results.csv"
    database_df.write_csv(csv_path)
    logging.info(f"Full results saved to {csv_path}")
    
    # Save as Excel if available
    try:
        excel_path = report_dir / "full_screening_results.xlsx"
        database_df.write_excel(excel_path)
        logging.info(f"Full results also saved to {excel_path}")
    except Exception as e:
        logging.warning(f"Could not save Excel file: {e}")
    
    # Save as JSON for programmatic access
    try:
        json_path = report_dir / "full_screening_results.json"
        with open(json_path, 'w') as f:
            json.dump(database_df.to_pandas().to_dict(orient='records'), f, indent=2)
        logging.info(f"Full results also saved to {json_path}")
    except Exception as e:
        logging.warning(f"Could not save JSON file: {e}")

    # Convert to pandas once for all plotting operations
    df_pandas = database_df.to_pandas()
    
    # Check if we have valid data for plotting
    if df_pandas.empty:
        logging.warning("Pandas dataframe is empty, skipping plots")
        return
        
    # Check if docking_score column exists and has valid data
    if 'docking_score' not in df_pandas.columns:
        logging.warning("'docking_score' column not found in dataframe")
        return
        
    # Remove any rows with NaN docking scores
    df_pandas = df_pandas.dropna(subset=['docking_score'])
    if df_pandas.empty:
        logging.warning("No valid docking scores found after removing NaN values")
        return
        
    logging.info(f"Plotting data shape: {df_pandas.shape}")
    logging.info(f"Docking score range: {df_pandas['docking_score'].min()} to {df_pandas['docking_score'].max()}")
    
    n_samples = len(df_pandas)
    score_variance = df_pandas['docking_score'].var()
    
    # Plot Score Distributions - Handle different cases
    try:
        plt.figure(figsize=(12, 7))
        
        if n_samples == 1:
            # Single data point - create a bar plot
            plt.bar([df_pandas['docking_score'].iloc[0]], [1], width=0.1, alpha=0.7)
            plt.ylabel("Count")
            plt.title("Single Docking Score Result")
        elif score_variance == 0 or n_samples < 3:
            # No variance or very few points - create histogram instead of KDE
            plt.hist(df_pandas['docking_score'], bins=max(1, n_samples), alpha=0.7, edgecolor='black')
            plt.ylabel("Count")
            plt.title(f"Distribution of {n_samples} Docking Score(s)")
        else:
            # Enough variance for KDE plot
            if 'docking_tool' in df_pandas.columns and df_pandas['docking_tool'].nunique() > 1:
                sns.kdeplot(data=df_pandas, x='docking_score', hue='docking_tool', fill=True, common_norm=False)
            else:
                sns.kdeplot(data=df_pandas, x='docking_score', fill=True)
            plt.ylabel("Density")
            plt.title("Distribution of Primary Docking Scores")
            
        plt.xlabel("Docking Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = plots_dir / "docking_score_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Score distribution plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to create score distribution plot: {e}")
        plt.close()  # Ensure figure is closed even if error occurs

    # Plot Correlation Matrix - Handle different cases
    score_cols = [col for col in df_pandas.columns if 'score' in col]
    if len(score_cols) > 1:
        try:
            plt.figure(figsize=(10, 8))
            # Select only numeric score columns and remove any rows with NaN
            score_data = df_pandas[score_cols].select_dtypes(include=[np.number]).dropna()
            
            if score_data.empty:
                logging.warning("No valid score data for correlation matrix")
            elif score_data.shape[1] < 2:
                logging.warning("Need at least 2 score columns for correlation matrix")
            elif score_data.shape[0] < 2:
                # Not enough samples for correlation - show a table instead
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.axis('tight')
                ax.axis('off')
                table_data = score_data.round(3).values
                table = ax.table(cellText=table_data, 
                               colLabels=score_data.columns,
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 1.5)
                plt.title(f"Score Comparison ({n_samples} sample(s))")
            else:
                # Enough data for correlation matrix
                corr = score_data.corr()
                
                # Check if correlation matrix has valid values
                if corr.isna().all().all():
                    logging.warning("Correlation matrix contains only NaN values")
                    # Create a simple score comparison plot instead
                    plt.scatter(score_data.iloc[:, 0], score_data.iloc[:, 1], alpha=0.7, s=100)
                    plt.xlabel(score_data.columns[0])
                    plt.ylabel(score_data.columns[1])
                    plt.title("Score Comparison Scatter Plot")
                    plt.grid(True, alpha=0.3)
                else:
                    # Valid correlation matrix
                    mask = corr.isna()
                    sns.heatmap(corr, annot=True, cmap='viridis', fmt=".3f", mask=mask,
                              square=True, linewidths=0.5)
                    plt.title("Correlation Matrix of Scoring Functions")
            
            plt.tight_layout()
            plot_path = plots_dir / "score_correlation_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Score analysis plot saved to {plot_path}")
            
        except Exception as e:
            logging.error(f"Failed to create score analysis plot: {e}")
            plt.close()  # Ensure figure is closed even if error occurs
    else:
        logging.info(f"Only {len(score_cols)} score column(s) found, skipping correlation matrix")

    # Write a final report
    report_path = report_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write("--- Virtual Screening Summary Report ---\n\n")
        f.write(f"Total ligands processed: {database_df.shape[0]}\n")
        f.write(f"Docking tools used: {database_df['docking_tool'].unique().to_list()}\n\n")
        
        f.write("Top 5 Overall Hits (based on primary docking score):\n")
        # Sorting depends on the tool, so we show top from each
        for tool in database_df['docking_tool'].unique():
            is_desc = tool != 'Autodock-Vina'
            f.write(f"\n--- Top hits from {tool} ---\n")
            top_hits = database_df.filter(pl.col('docking_tool') == tool).sort('docking_score', descending=is_desc).head(5)
            f.write(top_hits.to_pandas().to_string())
            f.write("\n")
            
        f.write("\n\n--- Analysis & Interpretation ---\n")
        f.write("1. Docking Score Distribution Plot: This plot shows how well each docking tool separated the scores. A wider distribution is often preferable as it indicates better discrimination between potential binders.\n\n")
        f.write("2. Correlation Matrix Plot: This shows how similarly the different scoring functions rank the compounds. A low correlation between a classical score (like Vina) and an ML score (like Gnina) can be valuable, as it suggests they capture different aspects of binding. A compound that scores well on two uncorrelated functions is a strong candidate.\n\n")
        f.write("3. Final Results CSV: The file 'full_screening_results.csv' contains all raw data for your own analysis.\n")
    logging.info(f"Summary report saved to {report_path}")

def main():
    """Main function to orchestrate the virtual screening workflow."""
    parser = argparse.ArgumentParser(description="Command-Line Virtual Screening Platform")
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default="screening_results",
        help="The directory to save all results."
    )
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    create_directories(output_path)
    
    # --- Step 0: Get User Preferences ---
    get_user_preferences()

    # --- Step 1: Get Protein ---
    protein_pdb_raw = get_protein_input(output_path / "protein")
    if not protein_pdb_raw:
        sys.exit(1)

    # --- Step 2: Prepare Protein ---
    protein_pdb_fixed, protein_pdbqt = prepare_protein(protein_pdb_raw, output_path / "protein")

    # --- Step 3: Get Ligands ---
    ligand_sdf_files = get_ligand_input(output_path / "ligands")
    if not ligand_sdf_files:
        logging.error("No ligands were provided or generated. Exiting.")
        sys.exit(1)

    # --- Step 4: Prepare Ligands ---
    ligand_pdbqt_files = prepare_ligands(ligand_sdf_files, output_path / "ligands")

    # --- Step 5: Run Docking ---
    docking_df = run_docking(
        protein_pdb_fixed, 
        protein_pdbqt, 
        ligand_pdbqt_files,
        ligand_sdf_files, # Pass original SDFs for DiffDock
        output_path
    )
    
    # --- Step 6: Run Rescoring ---
    final_df = run_rescoring(docking_df, protein_pdbqt, output_path)

    # --- Step 7: Analyze and Save ---
    analyze_and_save_results(final_df, output_path)
    run_plip_interactions(output_path)
    
    logging.info("--- Virtual Screening Pipeline Finished Successfully! ---")
    logging.info(f"All results are in the '{output_path.resolve()}' directory.")

if __name__ == "__main__":
    main()