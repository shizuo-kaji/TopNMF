## from name = "torch_topological", version = "0.1.7"
"""Cubical complex calculation module."""

import torch
from torch import nn
from torch_topological.nn import PersistenceInformation
import gudhi
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import networkx as nx

import warnings
warnings.filterwarnings("ignore")


class CubicalComplex(nn.Module):
    """Calculate cubical complex persistence diagrams.

    This module calculates 'differentiable' persistence diagrams for
    structured data, such as images. This is achieved by calculating
    a *cubical complex*.

    Cubical complexes are the natural choice for calculating topological
    features of highly-structured inputs. See [Rieck20a]_ for an example
    of how to apply such topological features in practice.

    References
    ----------
    .. [Rieck20a] B. Rieck et al., "Uncovering the Topology of
       Time-Varying fMRI Data Using Cubical Complex", *Advances in
       Neural Information Processing Systems 33*, pp. 6900--6912, 2020.
    """

    def __init__(self, superlevel=False, dim=None, mode="T"):
        """Initialise new module.

        Parameters
        ----------
        superlevel : bool
            Indicates whether to calculate topological features based on
            superlevel sets. By default, *sublevel set filtrations* are
            used.

        dim : int or `None`
            If set, describes dimension of input data. This is meant to
            be the dimension of an individual image **without** channel
            information, if any. The value of `dim` will change the way
            an input tensor is being handled: additional dimensions, if
            present, will be treated as batches or channels. If not set
            to an integer value, :func:`forward` will just *guess* what
            to do with an input (which should work in most cases).

            For example, when dealing with volume data, i.e. 3D tensors,
            set `dim=3` when instantiating the class. This will permit a
            seamless user experience with *both* batched and non-batched
            input data sets.
        """
        super().__init__()

        # TODO: This is handled somewhat inelegantly below. Might be
        # smarter to update.
        self.superlevel = superlevel
        self.dim = dim
        self.mode=mode

    def forward(self, x):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on a
        cubical complex and returning a set of persistence diagrams.
        The way the input will be interpreted depends on the presence
        of the `dim` attribute of this class. If `dim` is set, the
        *last* `dim` dimensions of an input tensor will be considered to
        contain the image data. If `dim` is not set, image dimensions
        will be guessed as follows:

        1. Tensor of dimension 2: a single image
        2. Tensor of dimension 3: a single 2D image with channels
        3. Tensor of dimension 4: a batch of 2D images with channels

        This is a conservative way of handling the data, ensuring that
        by default, 2D tensors with channel information and a potential
        batch information can be handled, since this is the default for
        many applications.

        To ensure that the class can handle e.g. 3D volume data, it is
        sufficient to set `dim = 3` when initialising the class. Refer
        to the examples and parameters sections for more details.

        Parameters
        ----------
        x : array_like
            Input image(s). If `dim` has not been set, will *guess* how
            to handle the input as follows: `x` can either be a 2D array
            of shape `(H, W)`, which is treated as a single image, or
            a 3D array/tensor of the form `(C, H, W)`, with `C`
            representing the number of channels, or a 4D array/tensor of
            the form `(B, C, H, W)`, with `B` being the batch size. If
            `dim` has been set, the same handling strategy applies, but
            the *last* `dim` dimensions of the tensor are being used for
            the cubical complex calculation. All subsequent dimensions
            will be assumed to represent batches or channels (in this
            order). Hence, if `dim` is set, the tensor must at most have
            `dim + 2` dimensions.

        Returns
        -------
        list of :class:`PersistenceInformation`
            List of :class:`PersistenceInformation`, containing both the
            persistence diagrams and the generators, i.e. the
            *pairings*, of a certain dimension of topological features.
            If `x` is a 3D array, returns a list of lists, in which the
            first dimension denotes the batch and the second dimension
            refers to the individual instances of
            :class:`PersistenceInformation` elements. Similar for
            higher-order tensors.

        Examples
        --------
        # Handling 3D tensors (volumes), either in batches or presented
        # individually to the function.
        >> cubical_complex = CubicalComplex(dim=3)
        >> cubical_complex(x)
        """
        # Dimension was provided; this makes calculating the *effective*
        # dimension of the tensor much easier: take everything but the
        # last `self.dim` dimensions.
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)

        # No dimension was provided; just use the shape provided by the
        # client.
        else:
            dims = len(x.shape) - 2

        # No additional dimensions present: a single image
        if dims == 0:
            return self._forward(x)

        # Handle image with channels, such as a tensor of the form `(C, H, W)`
        elif dims == 1:
            return [
                self._forward(x_) for x_ in x
            ]

        # Handle image with channels and batch index, such as a tensor of
        # the form `(B, C, H, W)`.
        elif dims == 2:
            return [
                    [self._forward(x__) for x__ in x_] for x_ in x
            ]

    def _forward(self, x):
        """Handle a single-channel image.

        This internal function handles the calculation of topological
        features for a single-channel image, i.e. an `array_like`.

        Parameters
        ----------
        x : array_like of shape `(d_1, d_2, ..., d_d)`
            Single-channel input image of arbitrary dimensions. Batch
            dimensions and channel dimensions have to to be handled by
            the calling function explicitly. This function interprets
            its input as a high-dimensional image.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        if self.superlevel:
            x = -x

        if self.mode=="T":
            cubical_complex = gudhi.CubicalComplex(
                dimensions=x.shape,
                top_dimensional_cells=x.flatten()
            )
        else:
            cubical_complex = gudhi.CubicalComplex(
                dimensions=x.shape,
                vertices=x.flatten()
            )

        # We need the persistence pairs first, even though we are *not*
        # using them directly here.
        cubical_complex.persistence()
        if self.mode=="T":
            cofaces = cubical_complex.cofaces_of_persistence_pairs()
        else:
            cofaces = cubical_complex.vertices_of_persistence_pairs()
        max_dim = len(x.shape)

        # TODO: Make this configurable; is it possible that users only
        # want to return a *part* of the data?
        persistence_information = [
            self._extract_generators_and_diagrams(
                x,
                cofaces,
                dim
            ) for dim in range(0, max_dim)
        ]

        return persistence_information

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            # 'Pair off' all the indices
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs, dim)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x, pairs, dim):

        xs = x.shape

        # Notice that `creators` and `destroyers` refer to pixel
        # coordinates in the image.
        creators = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 0], xs)
                ),
                dtype=torch.long
        )
        destroyers = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 1], xs)
                ),
                dtype=torch.long
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)))

        # TODO: Most efficient way to generate diagram again?
        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )




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




def pers_loss(D1, D2, pow=2, remove_longest_D1=True, remove_longest_D2=False):
    """Compute the persistence loss between two diagrams."""
    pers1 = torch.diff(D1, dim=1).reshape(-1)
    pers1, _ = torch.sort(pers1, dim=0)
    if remove_longest_D1:
        pers1 = pers1[:-1]

    pers2 = torch.diff(D2, dim=1).reshape(-1)
    pers2, _ = torch.sort(pers2, dim=0)
    if remove_longest_D2:
        pers2 = pers2[:-1]

    if len(pers1) > len(pers2):
        p = len(pers1) - len(pers2)
        loss = (pers1[:p]).pow(pow).sum()
        if len(pers2) > 0:
            loss += (pers1[p:] - pers2).abs().pow(pow).sum()
    else:
        p = len(pers2) - len(pers1)
        loss = (pers2[:p]).pow(pow).sum()
        if len(pers1) > 0:
            loss += (pers2[p:] - pers1).abs().pow(pow).sum()

    return loss



def weighted_persistence_loss(D, pow=2, eps=1e-4, remove_longest=True):
    """
    Compute weighted persistence loss:
    L_top(v) = (sum_i (|b_i - d_i|^p / d_i))^{1/p}

    Parameters:
    D : torch.Tensor
        Persistence diagram of shape [N, 2], where each row is (birth, death).
    pow : int
        The power p in the loss definition.
    eps : float
        Small constant to avoid division by zero when d_i is very small.
    remove_longest : bool
        Whether to remove the interval with longest persistence (optional, for stability).

    Returns:
    torch.Tensor
        The computed weighted topological loss.
    """

    births = D[:, 0]
    deaths = D[:, 1]
    persistence = (deaths - births).abs()
    
    # Optional: remove the longest bar
    if remove_longest and len(persistence) > 0:
        longest_idx = torch.argmax(persistence)
        mask = torch.ones(len(D), dtype=torch.bool)
        mask[longest_idx] = False
        births = births[mask]
        deaths = deaths[mask]
        persistence = persistence[mask]

    # Compute the weighted loss


    weights = (1.0 - torch.abs(deaths)).pow(pow)

    
    weighted = (weights * persistence).pow(pow)
    loss = weighted.sum()

    return loss


def plot_loss(losses, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    for key in losses.keys():
        ax.plot(losses[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    return ax


def plot_gallery(images, title="", n_col=5, n_row=5, cmap=plt.cm.gray, axs=None):
    if axs is None:
        fig,axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col*n_row)]):

        if n_row != 1:
            vmax = max(comp.max(), -comp.min())
            axs[i//n_col,i%n_col].imshow(comp, cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            axs[i//n_col,i%n_col].set_xticks(())
            axs[i//n_col,i%n_col].set_yticks(())


        else:
            vmax = max(comp.max(), -comp.min())
            axs[i%n_col].imshow(comp, cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            axs[i%n_col].set_xticks(())
            axs[i%n_col].set_yticks(())



def plot_time_series(images, title="", n_col=5, n_row=5, cmap=plt.cm.gray, axs=None):
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
        plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        if n_row != 1:
            ax = axs[i // n_col, i % n_col]
        else:
            ax = axs[i % n_col]

        ax.cla()  # ✅ 기존 선 제거
        ax.plot(comp)
        ax.set_xticks(())
        ax.set_yticks(())

    return axs



def plot_gallery_graph(edge_values, edge_index, title, n_col, n_row, axs):
    """
    edge_values: numpy.ndarray 또는 torch.Tensor, shape=(n_components, n_edges)
                 각 basis마다 엣지 가중치를 담은 행렬
    edge_index : torch.LongTensor of shape (2, n_edges)
                 모든 가능한 엣지를 (u,v) 쌍으로 담은 텐서 (각 열이 [u,v])
    title      : 그래프 제목 앞부분 (문자열)
    n_col, n_row: 서브플롯 그리드의 열/행 개수
    axs        : 미리 만든 figsize=(2*n_col, 2.26*n_row) 크기의 axes 배열 (shape=(n_row, n_col))
    """

    # 1) edge_index가 (2, n_edges) 형태이므로 “(n_edges, 2)” 모양으로 전치
    edge_index_pairs = edge_index.transpose(0, 1)  # shape = (n_edges, 2)

    # 2) 노드 위치(pos) 계산: 전체 노드를 포함하는 임시 그래프 생성
    #    모든 노드를 한 번에 넣고 spring_layout으로 좌표 구함
    nodes = set(int(u.item()) for u, v in edge_index_pairs) | set(int(v.item()) for u, v in edge_index_pairs)
    G_temp = nx.Graph()
    G_temp.add_nodes_from(nodes)
    # 엣지 리스트를 Python 리스트 형태로 변환
    G_temp.add_edges_from([(int(u.item()), int(v.item())) for u, v in edge_index_pairs])
    pos = nx.spring_layout(G_temp, seed=42)

    # 3) 실제 그릴 basis 수 = edge_values.shape[0]
    num_basis = edge_values.shape[0]
    max_plots = n_row * n_col

    # 4) “실제로 그릴 횟수”는 min(num_basis, max_plots)
    n_plots = min(num_basis, max_plots)

    # 5) 축/서브플롯 인덱스를 0 ~ (n_row*n_col-1)까지 미리 초기화해 두고
    #    필요 없는 나머지 축은 비워 두도록 처리
    for idx in range(max_plots):
        r = idx // n_col
        c = idx % n_col
        ax = axs[r, c]
        ax.clear()  # 일단 초기화

        if idx < n_plots:
            # idx 번째 basis를 그려야 하는 경우
            weights = edge_values[idx]  # shape = (n_edges,)

            # weights j번째가 0 이하이면 스킵
            for j, (u, v) in enumerate(edge_index_pairs):
                w = weights[j]
                if isinstance(w, torch.Tensor):
                    w = w.item()
                if w <= 0:
                    continue
                ux, uy = pos[int(u.item())]
                vx, vy = pos[int(v.item())]
                ax.plot(
                    [ux, vx],
                    [uy, vy],
                    color='tab:gray',
                    linewidth=w * 5
                )
            ax.set_title(f"{title} {idx}", fontsize=8)
            ax.set_xticks([])  # x축 눈금 없애기
            ax.set_yticks([])  # y축 눈금 없애기
        else:
            # idx >= n_plots라면, 이 축은 쓰지 않음 → 완전 빈칸으로 둔다
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    return axs




def plot_PD(images, n_col=5, n_row=5, axs=None, PHmode="V", superlevel=False):
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images[:(n_col*n_row)]):
        sign = -1 if superlevel else 1
        if PHmode == "V":
            cubical_complex = gudhi.CubicalComplex(vertices=sign * comp)
        else:
            cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=sign * comp)
        pd = cubical_complex.persistence()
        ax = axs[i // n_col, i % n_col]
        ax.clear()
        gudhi.plot_persistence_diagram(pd, axes=ax, legend=False, fontsize=4)
        ax.set_xticks(())
        ax.set_yticks(())



def plot_PD_graph(graphs, edge_list, n_col=5, n_row=5, axs=None, max_dim=1, superlevel=False):
    """
    graphs    : List of edge-weight 벡터 (각 항목은 길이 = len(all_edges))
    all_edges : 전체 가능한 엣지 리스트 [(u0,v0), (u1,v1), …]
    n_col, n_row: 한 페이지에 그릴 subplot 개수
    axs       : 이미 생성된 Axes 2D 배열 (없으면 새로 만듦)
    max_dim   : SimplexTree.expansion 할 최대 차원 (0-차원 PH만 쓰려면 1로 충분)
    superlevel: True이면 값에 부호 반전 적용
    """
    if axs is None:
        fig, axs = plt.subplots(n_row, n_col, figsize=(2. * n_col, 2.26 * n_row))
    axs = np.atleast_2d(axs)

    ph_model = GraphFiltrationPH(max_dim=max_dim, superlevel=superlevel)

    for i, edge_attr in enumerate(graphs[: (n_col * n_row)]):
        # 1) H₀ persistence 계산
        pers_info = ph_model(edge_list, edge_attr)

        # 2) 플로팅
        ax = axs[i // n_col, i % n_col]
        ax.clear()
        for pi in pers_info:
            dim = int(getattr(pi, "dimension", 0))
            # superlevel=True면 부호를 뒤집어 플롯
            pts = pi.diagram.detach().cpu().numpy()
            # gd_list = [(dim, (float(b), float(d))) for b, d in pts]

            tol = 1e-6
            # pts: shape (N, 2) = [[b, d], ...]
            m = np.isfinite(pts).all(axis=1) & (abs(pts[:, 1] - pts[:, 0]) > tol)
            arr = pts[m]
            gd_list = [(dim, (np.abs(d), np.abs(b))) for b, d in arr]

            gudhi.plot_persistence_diagram(gd_list, axes=ax, legend=False, fontsize=4, alpha=0.1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Graph {i}", fontsize=8)

    return axs





def weighted_persistence_loss(D, pow=2, eps=1e-4, remove_longest=True):
    births = D[:, 0]
    deaths = D[:, 1]
    persistence = (deaths - births).abs()

    if remove_longest and len(persistence) > 0:
        longest_idx = torch.argmax(persistence)
        mask = torch.ones(len(D), dtype=torch.bool)
        mask[longest_idx] = False
        births = births[mask]
        deaths = deaths[mask]
        persistence = persistence[mask]

    weights = (1.0 - torch.abs(deaths)).pow(pow)
    weighted = (weights * persistence).pow(pow)
    return weighted.sum()


def total_variation(v):
    return ((v[1:] - v[:-1])**2).sum()

