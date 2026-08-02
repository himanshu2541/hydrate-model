"""Typed exceptions for the hydrate equilibrium model.

Replaces bare `except Exception` catches that used to silently substitute
plausible-looking fallback values (a fabricated water activity, a zero
Langmuir constant) for a real failure. Physics code should raise one of
these instead of swallowing the error; only the per-T-point scan loop in
solvers/equilibrium.py catches them, and it records the failure rather than
inventing a number.
"""


class HydrateModelError(Exception):
    """Base class for all physics-layer failures in this package."""


class WaterActivityError(HydrateModelError):
    """Raised when the liquid-phase activity-coefficient calculation
    (modified UNIFAC) cannot produce a water activity."""


class LangmuirIntegrationError(HydrateModelError):
    """Raised when the Kihara cell-potential quadrature fails to converge."""


class ConvergenceError(HydrateModelError):
    """Raised when the equilibrium-pressure root-find fails to converge."""
