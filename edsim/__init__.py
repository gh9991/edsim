"""ED patient-flow simulation and prediction skeleton.

Design rule: everything downstream of a loader speaks the *canonical schema*
defined in `edsim.schema`. On hackathon day you write one new loader
(`edsim.loaders.portal`) and nothing else changes.
"""

__version__ = "0.1.0"
