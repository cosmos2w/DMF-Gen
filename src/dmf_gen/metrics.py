"""Physical metrics for changing-geometry reconstruction.

Example::

    error = support_relative_l2(target[:, 0], prediction[:, 0], support)
    shape = geometry_errors(support, prediction[:, mask_channel])

Field errors use the reference support only, so every method is scored on the
same points. The predicted support is thresholded at 0.5 before its error is
measured. The drag coefficient needs the optional ``physics`` dependencies.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

AIR_HEAT_CAPACITY_RATIO = 1.4
RELATIVE_ERROR_FLOOR_FRACTION = 0.15


def support_relative_l2(target: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> float:
    """``||prediction - target|| / ||target||`` over the reference support."""
    valid = np.asarray(support, dtype=bool)
    reference = np.asarray(target, dtype=np.float64)[valid]
    estimate = np.asarray(prediction, dtype=np.float64)[valid]
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-12))


def geometry_errors(
    support: np.ndarray, predicted_mask: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Relative L2 error and intersection-over-union of the predicted void or body.

    The prediction is thresholded, so the relative L2 error of the region
    indicator is ``sqrt(#disagreeing points / #reference region points)``.
    """
    region = ~np.asarray(support, dtype=bool)
    predicted_region = np.asarray(predicted_mask, dtype=np.float64) <= float(threshold)
    count = int(region.sum())
    if count == 0:
        return {"geometry_relative_l2": float("nan"), "geometry_iou": float("nan")}
    disagreement = int(np.logical_xor(region, predicted_region).sum())
    union = int(np.logical_or(region, predicted_region).sum())
    overlap = int(np.logical_and(region, predicted_region).sum())
    return {
        "geometry_relative_l2": float(np.sqrt(disagreement / count)),
        "geometry_iou": float(overlap / union),
    }


def stress_concentration(stress: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    """Return ``(K_t, sigma_max)`` with ``K_t = max / mean`` over the material.

    The mean over the material stands in for the nominal stress. ``K_t`` is
    scale-invariant, so read it together with the peak-stress error.
    """
    values = np.asarray(stress, dtype=np.float64)[np.asarray(support, dtype=bool)]
    peak = float(values.max())
    return peak / max(abs(float(values.mean())), 1e-12), peak


def airfoil_mach(
    density: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    pressure: np.ndarray,
    gamma: float = AIR_HEAT_CAPACITY_RATIO,
    floor: float = 1e-6,
) -> np.ndarray:
    """Mach number ``|(u, v)| / sqrt(gamma p / rho)`` of a primitive flow state."""
    rho = np.maximum(np.asarray(density, dtype=np.float64), floor)
    p = np.maximum(np.asarray(pressure, dtype=np.float64), floor)
    speed = np.hypot(np.asarray(velocity_x, np.float64), np.asarray(velocity_y, np.float64))
    return speed / np.sqrt(gamma * p / rho)


def _surface_force(x: np.ndarray, y: np.ndarray, pressure: np.ndarray) -> tuple[float, float]:
    """Pressure force per unit span on a closed polygon, with outward normals."""
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    p_mid = 0.5 * (pressure + np.roll(pressure, -1))
    dx, dy = x1 - x, y1 - y
    sign = 1.0 if float(np.sum(x * y1 - x1 * y)) >= 0 else -1.0
    return -float(np.sum(p_mid * sign * dy)), float(np.sum(p_mid * sign * dx))


def airfoil_drag_coefficients(
    target: np.ndarray,
    prediction: np.ndarray,
    unit_xy: np.ndarray,
    grid_shape: Sequence[int],
    physical_ranges: Sequence[Sequence[float]],
    channels: Mapping[str, int],
) -> tuple[float, float] | None:
    """Reference and predicted drag coefficients on the reference airfoil surface.

    The surface is the 0.5 contour of the reference mask, so both values share
    one integration path and their difference measures surface-pressure
    error. Pressure is interpolated linearly from fluid points to the surface,
    the force is resolved along the far-field flow direction from the grid
    border, and the reference length is the surface's streamwise extent in
    physical units. Returns ``None`` when the surface or freestream cannot be
    determined.
    """
    try:
        from matplotlib.figure import Figure
        from scipy.interpolate import griddata
    except ImportError as error:
        raise RuntimeError(
            "drag coefficients need pip install -e '.[physics]' (scipy and matplotlib)"
        ) from error

    ny, nx = (int(size) for size in grid_shape)
    reference = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    coords = np.asarray(unit_xy, dtype=np.float64)
    rho_i, u_i, v_i, p_i, mask_i = (channels[name] for name in ("rho", "u", "v", "p", "mask"))

    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border = border.ravel()
    rho_inf = float(np.mean(reference[border, rho_i]))
    u_inf = float(np.mean(reference[border, u_i]))
    v_inf = float(np.mean(reference[border, v_i]))
    speed = float(np.hypot(u_inf, v_inf))
    if speed < 1e-8 or rho_inf <= 0:
        return None
    cos_a, sin_a = u_inf / speed, v_inf / speed
    q_inf = 0.5 * rho_inf * speed**2
    p_inf = float(np.mean(reference[border, p_i]))

    # The manuscript contoured and interpolated in unit coordinates and then
    # scaled the surface; doing the same reproduces its drag values exactly.
    contours = Figure().subplots().contour(
        coords[:, 0].reshape(ny, nx),
        coords[:, 1].reshape(ny, nx),
        reference[:, mask_i].reshape(ny, nx),
        levels=[0.5],
    )
    segments = [segment for segment in contours.allsegs[0] if len(segment) >= 2]
    if not segments:
        return None
    surface = max(segments, key=lambda s: float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()))
    if surface.shape[0] < 8:
        return None
    if np.allclose(surface[0], surface[-1]):
        surface = surface[:-1]
    fluid = reference[:, mask_i] > 0.5
    p_true = griddata(coords[fluid], reference[fluid, p_i], surface, method="linear")
    p_pred = griddata(coords[fluid], estimate[fluid, p_i], surface, method="linear")
    finite = np.isfinite(p_true) & np.isfinite(p_pred)
    surface, p_true, p_pred = surface[finite], p_true[finite], p_pred[finite]
    if surface.shape[0] < 8:
        return None
    (x_low, x_high), (y_low, y_high) = physical_ranges
    x = x_low + (x_high - x_low) * surface[:, 0]
    y = y_low + (y_high - y_low) * surface[:, 1]
    length = float(x.max() - x.min())
    if length <= 0:
        return None

    def drag(surface_pressure: np.ndarray) -> float:
        force_x, force_y = _surface_force(x, y, surface_pressure - p_inf)
        return (force_x * cos_a + force_y * sin_a) / (q_inf * length)

    return drag(p_true), drag(p_pred)


def clamped_relative_errors(
    reference: Sequence[float],
    predicted: Sequence[float],
    floor_fraction: float = RELATIVE_ERROR_FLOOR_FRACTION,
) -> dict[str, float | int]:
    """Summarize per-case signed errors ``(pred - ref) / max(|ref|, floor)``.

    ``floor = floor_fraction * RMS(reference)`` over the cases; it only
    affects reference values near zero, such as near-zero drag.
    """
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(ref) & np.isfinite(pred)
    if not finite.any():
        return {"cases": 0}
    ref, pred = ref[finite], pred[finite]
    floor = float(floor_fraction) * float(np.sqrt(np.mean(ref**2)))
    errors = (pred - ref) / np.maximum(np.abs(ref), floor)
    return {
        "cases": int(ref.size),
        "mean_absolute_relative_error": float(np.mean(np.abs(errors))),
        "mean_signed_relative_error": float(np.mean(errors)),
        "median_signed_relative_error": float(np.median(errors)),
        "denominator_floor": floor,
        "clamped_cases": int(np.sum(np.abs(ref) < floor)),
    }


__all__ = [
    "AIR_HEAT_CAPACITY_RATIO",
    "RELATIVE_ERROR_FLOOR_FRACTION",
    "airfoil_drag_coefficients",
    "airfoil_mach",
    "clamped_relative_errors",
    "geometry_errors",
    "stress_concentration",
    "support_relative_l2",
]
