"""Fairness mitigation methods for clinical prediction models.

This module provides the mitigation base class and concrete implementations.
Each mitigator takes calibration-set data and applies group-specific
recalibration to test-set probabilities.

Design decision (v0.2): when multiple sensitive attributes are specified,
per-attribute recalibration produces *different* mitigated probabilities for
the same patient. This implementation returns per-attribute results and does
NOT combine them. The user (or the report) compares attribute-by-attribute.
A future version may support joint recalibration.
"""

from __future__ import annotations

import abc
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from isitfair.audit import _to_1d_array, _to_1d_float_array

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

_VALID_METHODS = ("isotonic", "platt", "beta")


class Mitigation(abc.ABC):
    """Abstract base class for fairness mitigation methods.

    Subclasses implement ``fit_transform`` which takes calibration data
    and returns mitigated test-set probabilities, one array per sensitive
    attribute.
    """

    @abc.abstractmethod
    def fit_transform(
        self,
        y_true_calib: Any,
        y_proba_calib: Any,
        y_proba_test: Any,
        sensitive_calib: dict[str, Any],
        sensitive_test: dict[str, Any],
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Fit on calibration data and transform test-set probabilities.

        Parameters
        ----------
        y_true_calib : array-like of shape (n_calib,)
            True binary labels for the calibration set.
        y_proba_calib : array-like of shape (n_calib,)
            Predicted probabilities for the calibration set.
        y_proba_test : array-like of shape (n_test,)
            Predicted probabilities for the test set.
        sensitive_calib : dict[str, array-like]
            Sensitive attribute arrays for the calibration set.
        sensitive_test : dict[str, array-like]
            Sensitive attribute arrays for the test set.

        Returns
        -------
        dict[str, NDArray]
            Mapping from attribute name to mitigated test-set probabilities.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# GroupRecalibration
# ---------------------------------------------------------------------------


class GroupRecalibration(Mitigation):
    """Group-specific recalibration of predicted probabilities.

    Fits a separate calibration model per subgroup of each sensitive
    attribute on the calibration set, then applies the per-subgroup
    mapping to the test set.

    Parameters
    ----------
    method : str
        Calibration method. One of ``"isotonic"`` (isotonic regression),
        ``"platt"`` (sigmoid / Platt scaling via logistic regression on
        the logit), or ``"beta"`` (beta calibration via the ``betacal``
        package).
    random_state : int | None, optional
        Seed for reproducibility (used by Platt scaling).

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported methods.
    ImportError
        If ``method="beta"`` and ``betacal`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> y = rng.binomial(1, 0.3, n).astype(float)
    >>> p = np.clip(y + rng.normal(0, 0.3, n), 0, 1)
    >>> sex = rng.choice(["M", "F"], n)
    >>> m = GroupRecalibration(method="isotonic")
    >>> result = m.fit_transform(y, p, p, {"sex": sex}, {"sex": sex})
    >>> sorted(result.keys())
    ['sex']
    """

    def __init__(
        self,
        method: str = "isotonic",
        random_state: int | None = None,
    ) -> None:
        if method not in _VALID_METHODS:
            msg = f"`method` must be one of {_VALID_METHODS}, got {method!r}"
            raise ValueError(msg)
        self._method = method
        self._random_state = random_state

    @property
    def method(self) -> str:
        """The calibration method name."""
        return self._method

    def fit_transform(
        self,
        y_true_calib: Any,
        y_proba_calib: Any,
        y_proba_test: Any,
        sensitive_calib: dict[str, Any],
        sensitive_test: dict[str, Any],
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Fit per-subgroup calibrators and transform test probabilities.

        For each sensitive attribute, a separate calibration model is
        fitted per subgroup on the calibration set. Test observations
        are then recalibrated using the calibrator matching their
        subgroup membership.

        Parameters
        ----------
        y_true_calib : array-like of shape (n_calib,)
            True binary labels for the calibration set.
        y_proba_calib : array-like of shape (n_calib,)
            Predicted probabilities for the calibration set.
        y_proba_test : array-like of shape (n_test,)
            Predicted probabilities for the test set.
        sensitive_calib : dict[str, array-like]
            Sensitive attribute arrays for the calibration set.
        sensitive_test : dict[str, array-like]
            Sensitive attribute arrays for the test set. Must have
            the same keys as ``sensitive_calib``. All subgroup values
            in the test set must have been seen in the calibration set.

        Returns
        -------
        dict[str, NDArray]
            Mapping from attribute name to an array of mitigated
            test-set probabilities (same length as ``y_proba_test``).
            Each attribute produces an independent recalibration.

        Raises
        ------
        ValueError
            If keys mismatch, lengths are wrong, or test subgroups
            are unseen in calibration.
        """
        # --- validate inputs ---
        y_calib = _to_1d_float_array(y_true_calib, "y_true_calib")
        p_calib = _to_1d_float_array(y_proba_calib, "y_proba_calib")
        p_test = _to_1d_float_array(y_proba_test, "y_proba_test")
        n_calib = len(y_calib)
        n_test = len(p_test)

        if len(p_calib) != n_calib:
            msg = (
                f"`y_true_calib` and `y_proba_calib` must have the same "
                f"length, got {n_calib} and {len(p_calib)}"
            )
            raise ValueError(msg)

        if set(sensitive_calib.keys()) != set(sensitive_test.keys()):
            msg = (
                f"`sensitive_calib` and `sensitive_test` must have the same "
                f"keys, got {sorted(sensitive_calib.keys())} and "
                f"{sorted(sensitive_test.keys())}"
            )
            raise ValueError(msg)

        # --- coerce sensitive features ---
        calib_features: dict[str, npt.NDArray[Any]] = {}
        test_features: dict[str, npt.NDArray[Any]] = {}
        for attr_name in sensitive_calib:
            c_arr = _to_1d_array(
                sensitive_calib[attr_name],
                f"sensitive_calib['{attr_name}']",
            )
            if len(c_arr) != n_calib:
                msg = f"sensitive_calib['{attr_name}'] has length {len(c_arr)}, expected {n_calib}"
                raise ValueError(msg)
            calib_features[attr_name] = c_arr

            t_arr = _to_1d_array(
                sensitive_test[attr_name],
                f"sensitive_test['{attr_name}']",
            )
            if len(t_arr) != n_test:
                msg = f"sensitive_test['{attr_name}'] has length {len(t_arr)}, expected {n_test}"
                raise ValueError(msg)
            test_features[attr_name] = t_arr

        # --- per-attribute recalibration ---
        results: dict[str, npt.NDArray[np.floating[Any]]] = {}
        for attr_name in calib_features:
            c_groups = calib_features[attr_name]
            t_groups = test_features[attr_name]
            calib_labels = set(np.unique(c_groups))
            test_labels = set(np.unique(t_groups))

            unseen = test_labels - calib_labels
            if unseen:
                msg = (
                    f"sensitive_test['{attr_name}'] contains subgroups "
                    f"not seen in calibration: {sorted(unseen)}"
                )
                raise ValueError(msg)

            # Fit one calibrator per subgroup
            calibrators: dict[str, Any] = {}
            for group in sorted(calib_labels):
                mask = c_groups == group
                y_sub = y_calib[mask]
                p_sub = p_calib[mask]
                calibrators[str(group)] = self._fit_single(y_sub, p_sub)

            # Apply to test set
            mitigated = np.empty(n_test, dtype=np.float64)
            for group in sorted(test_labels):
                mask = t_groups == group
                p_sub = p_test[mask]
                mitigated[mask] = self._predict_single(
                    calibrators[str(group)],
                    p_sub,
                )

            results[attr_name] = np.clip(mitigated, 0.0, 1.0)

        return results

    def _fit_single(
        self,
        y_true: npt.NDArray[np.floating[Any]],
        y_proba: npt.NDArray[np.floating[Any]],
    ) -> Any:
        """Fit a single calibrator on one subgroup."""
        if self._method == "isotonic":
            model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            model.fit(y_proba, y_true)
            return model
        if self._method == "platt":
            lp = _safe_logit_for_platt(y_proba)
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=self._random_state,
            )
            model.fit(lp.reshape(-1, 1), y_true)
            return model
        # beta
        try:
            from betacal import BetaCalibration  # type: ignore[import-not-found]
        except ImportError:
            msg = (
                "beta calibration requires the `betacal` package. "
                "Install it with: pip install betacal"
            )
            raise ImportError(msg) from None
        model = BetaCalibration()
        model.fit(y_proba, y_true)
        return model

    def _predict_single(
        self,
        model: Any,
        y_proba: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply a fitted calibrator to produce mitigated probabilities."""
        if self._method == "isotonic":
            result: npt.NDArray[np.floating[Any]] = np.asarray(
                model.predict(y_proba),
                dtype=np.float64,
            )
            return result
        if self._method == "platt":
            lp = _safe_logit_for_platt(y_proba)
            result = np.asarray(
                model.predict_proba(lp.reshape(-1, 1))[:, 1],
                dtype=np.float64,
            )
            return result
        # beta
        result = np.asarray(model.predict(y_proba), dtype=np.float64)
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_PLATT_EPS = 1e-10


def _safe_logit_for_platt(
    y_proba: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Logit transform with clipping for Platt scaling."""
    from scipy.special import logit

    clipped: npt.NDArray[np.floating[Any]] = np.clip(
        y_proba,
        _PLATT_EPS,
        1 - _PLATT_EPS,
    )
    result: npt.NDArray[np.floating[Any]] = np.asarray(
        logit(clipped),
        dtype=np.float64,
    )
    return result


# ---------------------------------------------------------------------------
# WassersteinPostprocessing (formerly GroupThresholdOptimization)
# ---------------------------------------------------------------------------

_VALID_CONSTRAINTS = ("equalized_odds", "demographic_parity", "equal_opportunity")


class WassersteinPostprocessing(Mitigation):
    """Fair score post-processing via the 1-Wasserstein barycenter.

    Maps each subgroup's predicted-probability distribution to a common
    distribution (the 1-D Wasserstein barycenter of the per-group calibration
    distributions) using a per-group quantile-to-quantile transformation. The
    mitigated scores are real numbers in ``[0, 1]`` whose distribution is
    identical across subgroups (exact **demographic parity** on the
    calibration set), while the rank order within each subgroup is preserved
    by construction — so within-subgroup AUROC is exactly unchanged.

    Unlike Fairlearn's ``ThresholdOptimizer``, the output is a calibrated
    risk score, not a labeling decision, so downstream calibration and
    decision-curve analysis remain meaningful.

    Method
    ------
    For each sensitive group :math:`g`, let :math:`F_g` be the empirical CDF
    of the calibration-set predicted probabilities for that group, with
    quantile function :math:`F_g^{-1}`. The 1-D Wasserstein barycenter of the
    :math:`\\{F_g\\}` weighted by the group marginals :math:`\\pi_g` has
    quantile function

    .. math::

        Q_{\\mathrm{bary}}(u) \\;=\\; \\sum_g \\pi_g \\, F_g^{-1}(u),
        \\qquad u \\in [0, 1].

    For a test prediction :math:`p` in group :math:`g`, the mitigated score
    is :math:`T_g(p) = Q_{\\mathrm{bary}}(F_g(p))`. The map :math:`T_g` is
    monotone non-decreasing, so within-group rank order — and hence
    within-group AUROC — is preserved exactly.

    Parameters
    ----------
    constraint : str, default ``"demographic_parity"``
        Fairness constraint. Only ``"demographic_parity"`` is supported.
        Passing ``"equalized_odds"`` or ``"equal_opportunity"`` raises
        :class:`NotImplementedError`: by the impossibility theorem of
        Pleiss et al. (2017), a calibrated post-processing cannot enforce
        equalized odds beyond trivial cases.
    n_grid : int, default 1001
        Number of quantile-grid points used to discretise the barycenter
        quantile function. Larger values give a smoother transformation at
        marginal computational cost.
    random_state : int | None, optional
        Unused (kept for API parity with sklearn estimators); reserved for
        future randomised barycenter variants.

    References
    ----------
    Chzhen, E., Denis, C., Hebiri, M., Oneto, L., Pontil, M. (2020).
    *Fair Regression with Wasserstein Barycenters.* NeurIPS 2020.

    Jiang, R., Pacchiano, A., Stepleton, T., Jiang, H., Chiappa, S. (2020).
    *Wasserstein Fair Classification.* UAI 2020.

    Gordaliza, P., del Barrio, E., Gamboa, F., Loubes, J.-M. (2019).
    *Obtaining Fairness using Optimal Transport Theory.* ICML 2019.

    Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., Weinberger, K. Q.
    (2017). *On Fairness and Calibration.* NeurIPS 2017.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> y = rng.binomial(1, 0.3, n).astype(float)
    >>> p = np.clip(y + rng.normal(0, 0.3, n), 0, 1)
    >>> sex = rng.choice(["M", "F"], n)
    >>> m = WassersteinPostprocessing()
    >>> result = m.fit_transform(y, p, p, {"sex": sex}, {"sex": sex})
    >>> sorted(result.keys())
    ['sex']
    """

    def __init__(
        self,
        constraint: str = "demographic_parity",
        n_grid: int = 1001,
        random_state: int | None = None,
    ) -> None:
        if constraint not in _VALID_CONSTRAINTS:
            msg = f"`constraint` must be one of {_VALID_CONSTRAINTS}, got {constraint!r}"
            raise ValueError(msg)
        if constraint != "demographic_parity":
            msg = (
                f"WassersteinPostprocessing only supports "
                f"constraint='demographic_parity'; got {constraint!r}. "
                "Equalized odds / equal opportunity are precluded by the "
                "impossibility theorem of Pleiss et al. (2017) — a "
                "post-processing mapping cannot generally satisfy both "
                "calibration and equalized odds when group base rates "
                "differ."
            )
            raise NotImplementedError(msg)
        if n_grid < 2:
            msg = f"`n_grid` must be >= 2, got {n_grid}"
            raise ValueError(msg)
        self._constraint = constraint
        self._n_grid = int(n_grid)
        self._random_state = random_state

    @property
    def constraint(self) -> str:
        """The fairness constraint name."""
        return self._constraint

    def fit_transform(
        self,
        y_true_calib: Any,
        y_proba_calib: Any,
        y_proba_test: Any,
        sensitive_calib: dict[str, Any],
        sensitive_test: dict[str, Any],
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Fit the per-attribute Wasserstein barycenter and transform test scores.

        Parameters
        ----------
        y_true_calib : array-like of shape (n_calib,)
            True binary labels (unused — barycenter is on the score
            distribution only; the argument is kept for API parity with
            :class:`Mitigation`).
        y_proba_calib : array-like of shape (n_calib,)
            Predicted probabilities for the calibration set.
        y_proba_test : array-like of shape (n_test,)
            Predicted probabilities for the test set.
        sensitive_calib : dict[str, array-like]
            Sensitive attribute arrays for the calibration set.
        sensitive_test : dict[str, array-like]
            Sensitive attribute arrays for the test set. Must have the
            same keys as ``sensitive_calib``; test subgroups must have
            been seen in calibration.

        Returns
        -------
        dict[str, NDArray]
            Mapping from attribute name to mitigated test-set scores in
            ``[0, 1]``, one independent transformation per attribute.

        Raises
        ------
        ValueError
            If keys mismatch, lengths are wrong, or test subgroups are
            unseen in calibration.
        """
        # y_true_calib is unused but kept for Mitigation API parity
        _ = _to_1d_float_array(y_true_calib, "y_true_calib")
        p_calib = _to_1d_float_array(y_proba_calib, "y_proba_calib")
        p_test = _to_1d_float_array(y_proba_test, "y_proba_test")
        n_calib = len(p_calib)
        n_test = len(p_test)

        if set(sensitive_calib.keys()) != set(sensitive_test.keys()):
            msg = (
                f"`sensitive_calib` and `sensitive_test` must have the same "
                f"keys, got {sorted(sensitive_calib.keys())} and "
                f"{sorted(sensitive_test.keys())}"
            )
            raise ValueError(msg)

        u_grid: npt.NDArray[np.floating[Any]] = np.linspace(0.0, 1.0, self._n_grid)
        results: dict[str, npt.NDArray[np.floating[Any]]] = {}

        for attr_name in sensitive_calib:
            c_sens = _to_1d_array(
                sensitive_calib[attr_name],
                f"sensitive_calib['{attr_name}']",
            )
            if len(c_sens) != n_calib:
                msg = f"sensitive_calib['{attr_name}'] has length {len(c_sens)}, expected {n_calib}"
                raise ValueError(msg)

            t_sens = _to_1d_array(
                sensitive_test[attr_name],
                f"sensitive_test['{attr_name}']",
            )
            if len(t_sens) != n_test:
                msg = f"sensitive_test['{attr_name}'] has length {len(t_sens)}, expected {n_test}"
                raise ValueError(msg)

            calib_groups = np.unique(c_sens)
            unseen = set(np.unique(t_sens)) - set(calib_groups.tolist())
            if unseen:
                msg = (
                    f"sensitive_test['{attr_name}'] contains subgroups "
                    f"not seen in calibration: {sorted(unseen)}"
                )
                raise ValueError(msg)

            # Per-group sorted calibration scores and marginal weights.
            sorted_per_group: dict[Any, npt.NDArray[np.floating[Any]]] = {}
            pi_per_group: dict[Any, float] = {}
            for g in calib_groups:
                mask = c_sens == g
                p_g = np.sort(p_calib[mask])
                sorted_per_group[g] = p_g
                pi_per_group[g] = float(mask.sum()) / n_calib

            # Build the barycenter quantile function on a shared grid:
            # Q_bary(u) = sum_g pi_g * F_g^{-1}(u), where F_g^{-1}(u) is the
            # u-quantile of group g's calibration scores.
            q_bary = np.zeros_like(u_grid)
            for g, p_g in sorted_per_group.items():
                q_bary += pi_per_group[g] * np.quantile(p_g, u_grid)

            # Transform test scores per group: T_g(p) = Q_bary(F_g(p)).
            # F_g(p) is the empirical CDF of group g's calibration scores
            # evaluated at p. We use right-side searchsorted so that ties
            # map to the upper-empirical CDF value, consistent with
            # numpy.quantile's default.
            mitigated = np.empty(n_test, dtype=np.float64)
            for g in calib_groups:
                t_mask = t_sens == g
                if not t_mask.any():
                    continue
                p_g = sorted_per_group[g]
                n_g = len(p_g)
                if n_g == 0:
                    # Should be impossible given the unseen check, but be defensive.
                    mitigated[t_mask] = p_test[t_mask]
                    continue
                # Empirical CDF F_g(p) in [0, 1].
                rank = np.searchsorted(p_g, p_test[t_mask], side="right")
                u_g = rank / n_g
                # Linear interp onto the barycenter grid.
                mitigated[t_mask] = np.interp(u_g, u_grid, q_bary)

            results[attr_name] = np.clip(mitigated, 0.0, 1.0)

        return results


# Deprecated alias — the previous name was descriptive of Fairlearn's
# threshold-tuning approach. The implementation is now Wasserstein-barycenter
# score post-processing; the old name is kept for backward compatibility.
class GroupThresholdOptimization(WassersteinPostprocessing):
    """Deprecated alias for :class:`WassersteinPostprocessing`.

    The original implementation wrapped Fairlearn's ``ThresholdOptimizer``
    and returned a randomized-classifier labeling probability via
    ``_pmf_predict``. That output is bimodal at ``{0, 1}`` and is not a
    calibrated risk score, so downstream calibration / decision-curve
    analysis on the mitigated audit produced degenerate (AUROC ≈ 0.5,
    sensitivity ≈ 0) results.

    Use :class:`WassersteinPostprocessing` instead. The constraint
    ``"equalized_odds"`` and ``"equal_opportunity"`` are no longer
    supported (they were degenerate in practice and are precluded by
    Pleiss et al. 2017 for any post-processing that preserves calibration).
    """

    def __init__(
        self,
        constraint: str = "demographic_parity",
        n_grid: int = 1001,
        random_state: int | None = None,
    ) -> None:
        warnings.warn(
            "GroupThresholdOptimization is deprecated and renamed to "
            "WassersteinPostprocessing. The underlying implementation has "
            "switched from Fairlearn's ThresholdOptimizer to a Wasserstein "
            "barycenter score transformation (Chzhen et al. 2020). "
            "Use isitfair.WassersteinPostprocessing in new code.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(constraint=constraint, n_grid=n_grid, random_state=random_state)


# ---------------------------------------------------------------------------
# Reweighing
# ---------------------------------------------------------------------------


class Reweighing:
    """Kamiran-Calders reweighing for fair model training.

    Computes sample weights that equalize the label distribution across
    subgroups of a sensitive attribute. The user passes these weights to
    their own training pipeline (e.g. ``sample_weight`` in sklearn), then
    constructs a new ``FairnessAudit`` on the retrained predictions.

    This class does NOT inherit from ``Mitigation`` because reweighing
    requires model retraining — it cannot be applied post-hoc to existing
    predictions via ``FairnessAudit.mitigate()``.

    The weight formula (Kamiran & Calders 2012) is:

        w(g, y) = P(Y=y) * P(G=g) / P(Y=y, G=g)

    Parameters
    ----------
    random_state : int | None, optional
        Reserved for future use (currently unused).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from isitfair.mitigation import Reweighing
    >>> from isitfair.audit import FairnessAudit
    >>>
    >>> # Step 1: compute weights on training data
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> y_train = rng.binomial(1, 0.3, n).astype(float)
    >>> sex_train = rng.choice(["M", "F"], n)
    >>> X_train = rng.normal(size=(n, 5))
    >>>
    >>> rw = Reweighing()
    >>> weights = rw.compute_weights(y_true=y_train, sensitive=sex_train)
    >>>
    >>> # Step 2: train model with weights
    >>> model = LogisticRegression(random_state=0)
    >>> model.fit(X_train, y_train, sample_weight=weights)
    LogisticRegression(random_state=0)
    >>>
    >>> # Step 3: audit the retrained model on held-out data
    >>> # (use FairnessAudit on the new predictions as usual)
    """

    def __init__(self, random_state: int | None = None) -> None:
        self._random_state = random_state

    def compute_weights(
        self,
        y_true: Any,
        sensitive: Any,
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute Kamiran-Calders sample weights.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        sensitive : array-like of shape (n_samples,)
            Sensitive attribute labels (single attribute).

        Returns
        -------
        NDArray of shape (n_samples,)
            Per-sample weights. Positive-class members of underrepresented
            groups get higher weights; the product sums to ``n_samples``.

        Raises
        ------
        ValueError
            If inputs have wrong shape or labels are not binary.
        """
        y = _to_1d_float_array(y_true, "y_true")
        s = _to_1d_array(sensitive, "sensitive")
        n = len(y)

        if len(s) != n:
            msg = f"`y_true` and `sensitive` must have the same length, got {n} and {len(s)}"
            raise ValueError(msg)

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0.0, 1.0}):
            msg = f"`y_true` must contain only 0 and 1, got {sorted(unique_labels)}"
            raise ValueError(msg)

        groups = np.unique(s)
        weights = np.ones(n, dtype=np.float64)

        for g in groups:
            for label in [0.0, 1.0]:
                # P(Y=y)
                p_y = float(np.sum(y == label)) / n
                # P(G=g)
                p_g = float(np.sum(s == g)) / n
                # P(Y=y, G=g)
                joint_mask = (y == label) & (s == g)
                p_joint = float(np.sum(joint_mask)) / n
                if p_joint > 0:
                    weights[joint_mask] = p_y * p_g / p_joint

        return weights
