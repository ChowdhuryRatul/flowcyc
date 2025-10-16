# extract_ca_coordinates.py
# Usage: python extract_ca_coordinates.py input.pdb output.txt

import sys

if len(sys.argv) < 3:
    print("Usage: python extract_ca_coordinates.py input.pdb output.txt")
    sys.exit(1)

input_pdb = sys.argv[1]
output_file = sys.argv[2]

with open(input_pdb, 'r') as pdb, open(output_file, 'w') as out:
    for line in pdb:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            out.write(f"{res_name:>3} {chain_id:>1} {res_seq:>4} {x:8.3f} {y:8.3f} {z:8.3f}\n")

print(f"Cα coordinates written to {output_file}")

