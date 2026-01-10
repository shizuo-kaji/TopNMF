import torch
from torch import nn
from typing import List, Tuple, Union
import gudhi
from torch_topological.nn import PersistenceInformation


__all__ = ['GraphFiltrationPH']

class GraphFiltrationPH(nn.Module):
    def __init__(self, max_dim=1, superlevel: bool = False):
        super().__init__()
        self.max_dim = max_dim
        self.superlevel = superlevel

    def forward(self,
                all_edges: List[Tuple[int, int]],
                edge_weight: Union[torch.Tensor, List[float]]
               ) -> List["PersistenceInformation"]:

        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32)
        edge_weight.requires_grad_(True)

        edge_index = torch.tensor(all_edges, dtype=torch.long).T
        num_edges = edge_index.shape[1]
        num_nodes = int(edge_index.max().item()) + 1

        # 모든 vertex weight를 1.0으로 고정
        f_vertices = torch.full((num_nodes,), -1.0, dtype=torch.float32, device=edge_weight.device)
        f_vertices.requires_grad_(True)


        edge_weight_filt = -edge_weight if self.superlevel else edge_weight

        st = gudhi.SimplexTree()
        simplex_to_index = {}
        gid = 0

        for i in range(num_nodes):
            filtration_val = f_vertices[i].detach().cpu().item()
            st.insert([i], filtration=filtration_val)
            simplex_to_index[(i,)] = gid
            gid += 1

        # for simplex, filtration_value in st.get_filtration():
        #     print(f"Simplex: {simplex}, Filtration: {filtration_value}")

        for i, (u, v) in enumerate(all_edges):
            val = edge_weight_filt[i].detach().cpu().item()
            st.insert([u, v], filtration=val)
            simplex_to_index[tuple(sorted((u, v)))] = gid
            gid += 1

        st.make_filtration_non_decreasing()

        all_filtration_values = torch.tensor(
            [f for _, f in st.get_filtration()],
            dtype=torch.float32,
            device=f_vertices.device
        )

        st.compute_persistence(persistence_dim_max=True)

        # for simplex, filtration_value in st.get_filtration():
        #     print(f"Simplex: {simplex}, Filtration: {filtration_value}")


        results = []

        for dim in [0, 1]:
            diagram_entries = []
            pairing_entries = []

            for s1, s2 in st.persistence_pairs():
                # print(s1,s2)
                dim_s1 = len(s1) - 1 if s1 else -1
                dim_s2 = len(s2) - 1 if s2 else -1
                # feature_dim = max(dim_s1, dim_s2)
                feature_dim = len(s1) - 1

                if feature_dim != dim:
                    continue

                # === birth filtration ===
                if len(s1) == 1:
                    b = f_vertices[s1[0]]
                elif len(s1) == 2:
                    idx = all_edges.index(tuple(sorted(s1)))
                    b = edge_weight_filt[idx]
                else:
                    b = torch.tensor(st.filtration(s1), device=f_vertices.device)

                # === death filtration ===
                if s2 and len(s2) == 1:
                    d = f_vertices[s2[0]]
                    d_simplex = tuple(s2)
                elif s2 and len(s2) == 2:
                    try:
                        idx = all_edges.index(tuple(sorted(s2)))
                        d = edge_weight_filt[idx]
                    except ValueError:
                        d = all_filtration_values.max()
                    d_simplex = tuple(s2)
                else:
                    d = all_filtration_values.max()
                    d_simplex = None

                diagram_entries.append([b, d])
                pairing_entries.append([tuple(s1), d_simplex])

            if diagram_entries:
                diagram = torch.stack([torch.stack(p) for p in diagram_entries])
                pairing = pairing_entries
            else:
                diagram = torch.zeros((0, 2), dtype=torch.float32, device=f_vertices.device)
                pairing = []

            results.append(PersistenceInformation(pairing=pairing, diagram=diagram, dimension=dim))

        return results
