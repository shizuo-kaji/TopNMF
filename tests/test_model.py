"""Tests for the TopologicalNMF estimator."""

from __future__ import annotations

import pytest
import torch

from TopNMF.model import TopologicalNMF


class _FailingComplex:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Persistence complex should not be called")


class _PeriodicityComplex:
    def __call__(self, component: torch.Tensor) -> list[torch.Tensor]:
        score = component.reshape(-1)[0]
        persistence = score * (3.0 ** 0.5)
        empty = torch.empty((0, 2), device=component.device)
        diagram = torch.stack([torch.zeros_like(persistence), persistence]).reshape(1, 2)
        return [empty, diagram]


def test_transform_recomputes_W_for_new_X(np) -> None:
    matrix = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.9, 0.7, 0.3, 0.2],
            [0.2, 0.1, 0.8, 1.0],
        ],
        dtype=float,
    )
    model = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)
    model.fit(
        matrix,
        n_iterations=50,
        lambda_top=0.0,
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    new_X = matrix[:2]
    W = model.transform(new_X)
    assert W.shape == (2, 2)
    assert np.all(W >= 0)
    assert np.isfinite(W).all()

    with pytest.raises(ValueError):
        model.transform(np.zeros((2, 99)))


def test_fit_skips_periodicity_none_components(np) -> None:
    matrix = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.9, 0.7, 0.3, 0.2],
            [0.2, 0.1, 0.8, 1.0],
        ],
        dtype=float,
    )
    model = TopologicalNMF(
        n_components=1,
        random_state=0,
        complex=_FailingComplex(),
        use_embedding=False,
    )

    model.fit(
        matrix,
        n_iterations=1,
        lambda_top=1.0,
        target_periodicity=[None],
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    assert model.get_components().shape == (1, matrix.shape[1])


def test_periodicity_targets_match_components_by_sorted_rank() -> None:
    model = TopologicalNMF(
        n_components=2,
        complex=_PeriodicityComplex(),
        use_embedding=False,
    )
    model.V = torch.tensor([[0.2], [0.8]], dtype=torch.float, requires_grad=True)

    loss_ph, _, _, _ = model._compute_component_losses(
        epoch=0,
        lambda_top=1.0,
        start_epoch_topological=0,
        ph_complex=model.complex,
        embedder=None,
        complex_inputs=None,
        PH_dims=[1],
        target_diagrams=None,
        target_periodicity=[1.0, 0.0],
        target_sparsity=None,
    )

    assert loss_ph.item() == pytest.approx(0.04)


def test_ph_loss_fn_is_applied_without_target_diagrams(np) -> None:
    """``lambda_top`` must not be a silent no-op when only ``ph_loss_fn`` is set."""
    matrix = np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.9, 0.7, 0.3, 0.2],
            [0.2, 0.1, 0.8, 1.0],
        ],
        dtype=float,
    )
    calls: list[object] = []

    def counting_loss(diagrams, PH_dims, target_diagrams, device, **kwargs):
        calls.append(target_diagrams)
        return torch.tensor(1.5, device=device)

    model = TopologicalNMF(
        n_components=2,
        random_state=0,
        complex=_PeriodicityComplex(),
        ph_loss_fn=counting_loss,
        use_embedding=False,
    )
    model.fit(
        matrix,
        n_iterations=2,
        lambda_top=1.0,
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    assert calls, "ph_loss_fn was never called"
    assert all(target is None for target in calls)
    assert model.get_losses()["PH"] == pytest.approx([1.5, 1.5])


def test_no_topological_loss_without_ph_loss_fn_or_periodicity(np) -> None:
    matrix = np.array([[1.0, 0.8, 0.2], [0.2, 0.1, 0.9]], dtype=float)
    model = TopologicalNMF(
        n_components=2,
        random_state=0,
        complex=_FailingComplex(),
        use_embedding=False,
    )
    model.fit(
        matrix,
        n_iterations=2,
        lambda_top=1.0,
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    assert model.get_losses()["PH"] == pytest.approx([0.0, 0.0])


def test_target_diagrams_without_ph_loss_fn_raises(np) -> None:
    matrix = np.array([[1.0, 0.8, 0.2], [0.2, 0.1, 0.9]], dtype=float)
    model = TopologicalNMF(n_components=2, random_state=0, use_embedding=False)

    with pytest.raises(ValueError, match="ph_loss_fn"):
        model.fit(
            matrix,
            n_iterations=1,
            lambda_top=1.0,
            target_diagrams=[torch.empty((0, 2))],
            init_method="random",
            scheduler_cls=None,
            verbose=False,
        )


def test_total_variation_uses_data_shape_for_structured_samples(np) -> None:
    """TV on an image basis must not wrap around row boundaries."""
    from TopNMF.losses import total_variation

    # A component that is constant along rows: 2-D TV sees only column steps.
    component = torch.tensor([1.0, 1.0, 5.0, 5.0], dtype=torch.float)
    model = TopologicalNMF(n_components=1, data_shape=(2, 2), use_embedding=False)
    model.V = component.reshape(1, 4).clone().requires_grad_(True)

    _, _, _, loss_tv = model._compute_component_losses(
        epoch=0,
        lambda_top=0.0,
        start_epoch_topological=0,
        ph_complex=None,
        embedder=None,
        complex_inputs=None,
        PH_dims=[1],
        target_diagrams=None,
        target_periodicity=None,
        target_sparsity=None,
    )

    expected = total_variation(component.reshape(2, 2))
    assert loss_tv.item() == pytest.approx(expected.item())
    assert loss_tv.item() != pytest.approx(total_variation(component).item())


def test_transform_is_reproducible_with_random_state(np) -> None:
    matrix = np.array([[1.0, 0.8, 0.2], [0.2, 0.1, 0.9]], dtype=float)
    model = TopologicalNMF(n_components=2, random_state=7, use_embedding=False)
    model.fit(
        matrix,
        n_iterations=20,
        lambda_top=0.0,
        init_method="random",
        scheduler_cls=None,
        verbose=False,
    )

    assert np.allclose(model.transform(matrix), model.transform(matrix))


def test_periodicity_none_target_is_ranked_but_not_applied() -> None:
    model = TopologicalNMF(
        n_components=2,
        complex=_PeriodicityComplex(),
        use_embedding=False,
    )
    model.V = torch.tensor([[0.2], [0.8]], dtype=torch.float, requires_grad=True)

    loss_ph, _, _, _ = model._compute_component_losses(
        epoch=0,
        lambda_top=1.0,
        start_epoch_topological=0,
        ph_complex=model.complex,
        embedder=None,
        complex_inputs=None,
        PH_dims=[1],
        target_diagrams=None,
        target_periodicity=[1.0, None],
        target_sparsity=None,
    )

    assert loss_ph.item() == pytest.approx(0.02)
