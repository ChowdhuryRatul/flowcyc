import numpy as np

# 20 canonical amino acids (alphabetical or standard order)
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]

# map from residue name to index
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_amino_acids(names):
    """
    names : list or array of residue names (e.g. ['ARG','ILE','CYS',...])
    returns: numpy array of shape (N, 20)
    """
    N = len(names)
    one_hot = np.zeros((N, len(AMINO_ACIDS)), dtype=np.float32)
    for i, name in enumerate(names):
        idx = AA_TO_IDX.get(name.upper(), None)
        if idx is not None:
            one_hot[i, idx] = 1.0
        else:
            raise ValueError(f"Unknown residue name: {name}")
    return one_hot

if __name__ == "__main__":
    node_names = ["ARG", "ILE", "CYS", "PRO"]
    onehot = one_hot_encode_amino_acids(node_names)

    print(onehot.shape)   # (4, 20)
    print(onehot)
    #print(AMINO_ACIDS)    # column order for reference