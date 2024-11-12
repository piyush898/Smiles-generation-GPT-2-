from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def calculate_tanimoto_similarity_gpu(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is not None and mol2 is not None:
        # TODO: Consider using RDKit-GPU for GPU-accelerated fingerprint calculations
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

        # Calculate Tanimoto similarity on CPU
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
        return similarity
    else:
        return 0.0  # Return 0.0 if there's an issue with molecule conversion

def save_smiles_with_non_100_similarity(molecule_file, pdb_file):
    # Read SMILES from generated_molecules.txt
    with open(molecule_file, 'r') as file:
        molecule_smiles = [line.strip() for line in file if line.strip()]

    # Read and filter valid SMILES from pdb.txt
    with open(pdb_file, 'r') as file:
        pdb_smiles = [line.strip() for line in file if line.strip()]
        pdb_smiles = [smiles for smiles in pdb_smiles if Chem.MolFromSmiles(smiles) is not None]

    # List to store SMILES with less than 100% similarity
    non_100_similarity_smiles = []

    # Iterate through each SMILES from pdb.txt and check Tanimoto similarity with all SMILES from generated_molecules.txt
    for pdb_smile in pdb_smiles:
        # Check if all similarities are less than 1.0
        if all(calculate_tanimoto_similarity_gpu(pdb_smile, molecule_smile) < 1.0 for molecule_smile in molecule_smiles):
            print(f"No 100% Tanimoto similarity found for SMILES: {pdb_smile}")
            non_100_similarity_smiles.append(pdb_smile)

    # Save SMILES with less than 100% similarity to a new file named non_100_similarity_inhibitor_smiles.txt
    with open('non_100_similarity_inhibitor_smiles.txt', 'w') as output_file:
        for smile in non_100_similarity_smiles:
            output_file.write(smile + '\n')

if __name__ == "__main__":
    # Replace 'generated_molecules.txt' and 'pdb.txt' with the actual file paths
    molecule_file_path = '/home/piyush22194/RNN/smiles-gpt-master/smiles-gpt-master/notebooks/generated_molecules.txt'
    pdb_file_path = '/home/piyush22194/biomdel master/BIMODAL-master/preprocessing/smiles_to_pdb/inhibitor_novel.txt'

    # Find and save SMILES with less than 100% similarity
    save_smiles_with_non_100_similarity(molecule_file_path, pdb_file_path)
