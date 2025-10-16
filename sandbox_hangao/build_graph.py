# build_pyg_graphs.py
import numpy as np
import torch
from torch_geometric.data import Data
from read_framew import load_snapshots_array
from adhoc_name_embedding import one_hot_encode_amino_acids
import torch
import pdb
# ----- your functions (paste/import these or keep them in the same file) -----
# load_snapshots_array(path, nodes_per_snapshot) -> {"coords": (S,N,3), "node_name": (N,), "node_id": (N,)}
# one_hot_encode_amino_acids(names) -> (N,20)

# Optional: try to use PyG helpers for edges if available
try:
    from torch_geometric.nn import radius_graph, knn_graph
    _HAS_PYG_EDGE_HELPERS = True
except Exception:
    _HAS_PYG_EDGE_HELPERS = False


def pairwise_edges_radius(pos: torch.Tensor, r: float) -> torch.Tensor:
    """
    Build undirected edges under radius r (Å) if PyG helpers are unavailable.
    pos: (N,3) float tensor
    returns edge_index: (2, E) long tensor, symmetric with no self-loops
    """
    N = pos.size(0)
    # Compute squared distances (N,N) efficiently
    # d^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = (pos**2).sum(dim=1, keepdim=True)              # (N,1)
    dist2 = x2 + x2.t() - 2 * (pos @ pos.t())           # (N,N)
    r2 = r * r
    mask = (dist2 <= r2) & (~torch.eye(N, dtype=torch.bool, device=pos.device))
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)  # directed both ways
    return edge_index


def pairwise_edges_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build k-NN edges (undirected, no self-loops) if PyG helpers are unavailable.
    """
    N = pos.size(0)
    # distances (N,N)
    x2 = (pos**2).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.t() - 2 * (pos @ pos.t())
    # set diagonal to +inf so it won't be picked
    dist2.fill_diagonal_(float('inf'))
    # top-k smallest distances along rows
    _, idx = torch.topk(dist2, k, largest=False, dim=1)  # (N,k)
    # make directed edges i->idx[i,j]
    src = torch.arange(N, device=pos.device).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    e = torch.stack([src, dst], dim=0)
    # symmetrize
    e_sym = torch.cat([e, torch.flip(e, [0])], dim=1)
    # remove duplicates
    e_sym = torch.unique(e_sym, dim=1)
    return e_sym


def make_edge_index(pos: torch.Tensor, method: str = "radius", radius: float = 8.0, k: int = 8) -> torch.Tensor:
    """
    Wrapper that selects radius/kNN and tries to use PyG's optimized ops.
    """
    if method == "radius":
        if _HAS_PYG_EDGE_HELPERS:
            # radius_graph returns directed edges; we’ll symmetrize
            e = radius_graph(pos, r=radius, loop=False)  # (2,E)
        else:
            e = pairwise_edges_radius(pos, radius)
    elif method == "knn":
        if _HAS_PYG_EDGE_HELPERS:
            e = knn_graph(pos, k=k, loop=False)  # directed kNN
        else:
            e = pairwise_edges_knn(pos, k)
    else:
        raise ValueError("method must be 'radius' or 'knn'.")

    # Symmetrize and remove duplicates for safety
    e_sym = torch.cat([e, torch.flip(e, [0])], dim=1)
    e_sym = torch.unique(e_sym, dim=1)
    return e_sym


def compute_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor):
    """
    pos: (N,3)
    edge_index: (2,E)
    Returns:
      - edge_vec: (E,3) relative vectors (x_j - x_i)
      - edge_len: (E,1) Euclidean distances
    """
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]                     # (E,3)
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)  # (E,1)
    return edge_vec, edge_len


def build_pyg_graphs_from_snapshots(
    coords_np: np.ndarray,           # (S,N,3)
    node_names_np: np.ndarray,       # (N,)
    node_ids_np: np.ndarray,         # (N,)
    edge_method: str = "radius",     # "radius" or "knn"
    radius: float = 8.0,             # Å (typical for Cα contact graph)
    k: int = 8,                      # kNN neighbors if using kNN
    add_node_id_feature: bool = False
):
    """
    Returns a list[Data], one graph per snapshot.
    - x: (N, F) node features (one-hot, optionally with node_id)
    - pos: (N, 3) coordinates
    - edge_index: (2, E) edges
    - edge_attr: (E, 4) [dx, dy, dz, dist]
    - node_id: (N,) long (as a separate field for reference)
    - snapshot_id: int (python attribute)
    """
    # one-hot features (N,20), constant across snapshots
    onehot = one_hot_encode_amino_acids(list(node_names_np))  # (N,20)
    x_base = torch.tensor(onehot, dtype=torch.float32)

    # Optional: attach node_id as a feature (scaled) or keep separately
    if add_node_id_feature:
        nid = torch.tensor(node_ids_np, dtype=torch.float32).unsqueeze(1)
        x_all = torch.cat([x_base, nid], dim=1)  # (N,21)
    else:
        x_all = x_base

    node_id_t = torch.tensor(node_ids_np, dtype=torch.long)

    S, N, _ = coords_np.shape
    graphs = []

    for s in range(S):
        pos = torch.tensor(coords_np[s], dtype=torch.float32)  # (N,3)
        edge_index = make_edge_index(pos, method=edge_method, radius=radius, k=k)
        edge_vec, edge_len = compute_edge_attr(pos, edge_index)
        edge_attr = torch.cat([edge_vec, edge_len], dim=1)     # (E,4)

        data = Data(
            x=x_all,                 # (N, F)
            pos=pos,                 # (N, 3)
            edge_index=edge_index,   # (2, E)
            edge_attr=edge_attr,     # (E, 4)
            node_id=node_id_t        # (N,)
        )
        data.snapshot_id = int(s)
        graphs.append(data)

    return graphs


# -------------------- Example usage --------------------
if __name__ == "__main__":
    # 1) Load snapshots (your function)
    data = load_snapshots_array("testoutput.txt", nodes_per_snapshot=28)
    coords = data["coords"]        # (S,N,3)
    node_name = data["node_name"]  # (N,)
    node_id = data["node_id"]      # (N,)
    # 2) Build graphs (radius graph at 8 Å)
    graphs = build_pyg_graphs_from_snapshots(
        coords_np=coords,
        node_names_np=node_name,
        node_ids_np=node_id,
        edge_method="radius",   # or "knn"
        radius=8.0,
        k=8,
        add_node_id_feature=False
    )
    pdb.set_trace()
    
    print(f"Built {len(graphs)} graphs; first graph:", graphs[0])
    print("x shape:", graphs[0].x.shape, "pos shape:", graphs[0].pos.shape,
          "edges:", graphs[0].edge_index.shape[1], "edge_attr:", graphs[0].edge_attr.shape)

    # If you want a PyG DataLoader later:
    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(graphs, batch_size=4, shuffle=True)
    # for batch in loader:
    #     print(batch)