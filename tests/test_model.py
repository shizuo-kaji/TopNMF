"""Tests for the TopologicalNMF estimator."""

from __future__ import annotations

from TopNMF.model import TopologicalNMF


class _FailingComplex:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("Persistence complex should not be called")


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
