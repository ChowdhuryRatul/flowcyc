import numpy as np
import pdb

def load_snapshots_array(path, nodes_per_snapshot):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append(parts)
    N = nodes_per_snapshot
    S = len(rows) // N
    node_name = np.array([r[0] for r in rows[:N]])
    node_id   = np.array([int(r[1]) for r in rows[:N]])
    coords = np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])
    coords = coords.reshape(S, N, 3)
    return {"coords": coords, "node_name": node_name, "node_id": node_id}

if __name__ == "__main__":
    data = load_snapshots_array("testoutput.txt", 28)
    print(data["coords"].shape)  # (num_snapshots, num_nodes, 3)
    pdb.set_trace()