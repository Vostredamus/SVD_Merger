"""
Direct SVD coordinate steering helpers.

Provides functions to resolve a steering mode (delta, target, multiplier)
to an exact recipe adjustment, and to generate coefficient table data.

The three modes:
  'delta'      : target = current + delta_value
  'target'     : target = target_value (direct)
  'multiplier' : target = current * multiplier (dimensionless scale)
"""

import numpy as np

_MODES = ("delta", "target", "multiplier")
_EPS = 1e-12


def resolve_steering(current_coeff: float, mode: str, value: float) -> float:
    """Return the resolved target coefficient for the given mode and value."""
    if mode not in _MODES:
        raise ValueError(f"Unknown steering mode: {mode}")
    current = float(current_coeff)
    value = float(value)
    if mode == "delta":
        return current + value
    if mode == "target":
        return value
    return current * value


def steering_recipe_adjustment(
    component_1indexed: int,
    current_coeff: float,
    mode: str,
    value: float,
) -> dict | None:
    """
    Return a recipe adjustment dict or None if no change.

    Uses target_coefficient (not donor/gamma) for direct steering.
    """
    target = resolve_steering(current_coeff, mode, value)
    if abs(target - float(current_coeff)) <= _EPS:
        return None
    return {
        "component": int(component_1indexed),
        "target_coefficient": float(target),
    }


def coefficient_table(analysis: dict, departure_label: str, overrides: dict) -> list:
    """
    Build a table showing starting and proposed coefficient for each component.

    overrides maps component index (1-based) to proposed target.
    """
    labels = analysis["labels"]
    p = labels.index(departure_label)
    coeffs = np.asarray(analysis["model_coefficients"], dtype=np.float64)[p]
    n_components = int(analysis["n_components"])
    rows = []
    for k in range(n_components):
        component = k + 1
        start = float(coeffs[k])
        proposed = float(overrides.get(component, start))
        rows.append(
            {
                "component": component,
                "starting_coefficient": start,
                "proposed_coefficient": proposed,
                "delta": proposed - start,
                "changed": abs(proposed - start) > _EPS,
            }
        )
    return rows
