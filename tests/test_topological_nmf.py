from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _DummyDiagram:
    diagram: object


class _RecordingComplex:
    def __init__(self, torch_module):
        self._torch = torch_module
        self.calls = []

    def __call__(self, *args):
        values = args[-1]
        self.calls.append(values.shape)
        device = values.device
        dtype = values.dtype
        diagrams = [
            _DummyDiagram(self._torch.tensor([[0.0, 0.5]], device=device, dtype=dtype)),
            _DummyDiagram(
                self._torch.tensor(
                    [[0.2, 0.9], [0.1, 0.4]],
                    device=device,
                    dtype=dtype,
                )
            ),
        ]
        return diagrams


def test_initialize_factors_random_is_reproducible(topological_nmf_module, np):
    x_matrix = np.abs(np.random.default_rng(0).normal(size=(6, 5)))

    model_a = topological_nmf_module.TopologicalNMF(
        n_components=3,
        random_state=123,
    )
    model_b = topological_nmf_module.TopologicalNMF(
        n_components=3,
        random_state=123,
    )

    w_a, v_a = model_a.initialize_factors(x_matrix, method="random")
    w_b, v_b = model_b.initialize_factors(x_matrix, method="random")

    assert w_a.shape == (6, 3)
    assert v_a.shape == (3, 5)
    assert np.all(w_a >= 0.0)
    assert np.all(v_a >= 0.0)
    np.testing.assert_allclose(w_a, w_b)
    np.testing.assert_allclose(v_a, v_b)


def test_unfitted_methods_raise(topological_nmf_module, np):
    model = topological_nmf_module.TopologicalNMF(n_components=2)

    with pytest.raises(ValueError, match="fitted"):
        model.transform(np.ones((2, 2), dtype=float))
    with pytest.raises(ValueError, match="fitted"):
        model.inverse_transform()
    with pytest.raises(ValueError, match="fitted"):
        model.get_components()


def test_fit_runs_and_records_losses(topological_nmf_module, np):
    x_matrix = np.abs(np.random.default_rng(1).normal(size=(8, 6)))
    model = topological_nmf_module.TopologicalNMF(
        n_components=2,
        random_state=7,
        complex=lambda *_: None,
    )

    fitted = model.fit(
        x_matrix,
        n_iterations=3,
        lr=0.01,
        lambda_top=0.0,
        mu_iter=1,
        W_iter=1,
        gd_iter=1,
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    assert fitted is model
    assert model.W.shape == (8, 2)
    assert model.V.shape == (2, 6)
    assert float(model.W.min()) >= 0.0
    assert float(model.V.min()) >= 0.0

    losses = model.get_losses()
    assert set(losses) == {"PH", "approx", "sparse_W", "sparse_V", "lr"}
    assert len(losses["approx"]) == 3
    assert len(losses["PH"]) == 3
    assert len(losses["lr"]) == 3

    transformed = model.transform(x_matrix)
    reconstructed = model.inverse_transform()
    reconstructed_from_external = model.inverse_transform(np.ones((3, 2), dtype=float))

    assert transformed.shape == (8, 2)
    assert reconstructed.shape == (8, 6)
    assert reconstructed_from_external.shape == (3, 6)


def test_fit_uses_data_shape_branch_with_custom_topological_loss(
    topological_nmf_module,
    np,
    torch,
):
    recorder = _RecordingComplex(torch)

    def custom_ph_loss(diagrams, ph_dims, target_diagrams, device, scale=1.0):
        total = torch.tensor(0.0, device=device)
        for dim in ph_dims:
            total = total + diagrams[dim].diagram[:, 1].sum()
        return scale * total

    x_matrix = np.abs(np.random.default_rng(2).normal(size=(5, 6)))
    model = topological_nmf_module.TopologicalNMF(
        n_components=2,
        random_state=11,
        complex=recorder,
        ph_loss_fn=custom_ph_loss,
        ph_loss_params={"scale": 0.1},
        data_shape=(2, 3),
        use_embedding=False,
    )

    target_diagrams = [
        torch.tensor([[0.0, 0.4]], dtype=torch.float32),
        torch.tensor([[0.1, 0.6]], dtype=torch.float32),
    ]
    model.fit(
        x_matrix,
        n_iterations=1,
        lr=0.01,
        lambda_top=1.0,
        mu_iter=0,
        gd_iter=1,
        init_method="random",
        scheduler_cls=None,
        target_diagrams=target_diagrams,
        target_periodicity=[0.25],
        PH_dims=[1],
        verbose=False,
    )

    assert recorder.calls == [torch.Size([2, 3]), torch.Size([2, 3])]
    assert len(model.losses["PH"]) == 1
    assert model.losses["PH"][0] > 0.0


def test_fit_uses_embedding_branch(topological_nmf_module, np, torch):
    recorder = _RecordingComplex(torch)
    x_matrix = np.abs(np.random.default_rng(3).normal(size=(4, 12)))
    model = topological_nmf_module.TopologicalNMF(
        n_components=1,
        random_state=5,
        complex=recorder,
        use_embedding=True,
    )

    model.fit(
        x_matrix,
        n_iterations=1,
        lr=0.01,
        lambda_top=1.0,
        mu_iter=0,
        gd_iter=1,
        init_method="random",
        scheduler_cls=None,
        target_periodicity=0.5,
        PH_dims=[1],
        M=2,
        tau=2,
        verbose=False,
    )

    assert len(recorder.calls) == 1
    assert recorder.calls[0][1] == 3
