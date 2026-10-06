"""
Automated ground-truth labelling for the confidence-filtering pipeline.

Replaces the manual CloudCompare loop (register -> C2C -> delete unseen areas ->
register again -> C2C again) with a scripted, rerunnable stage driven by one
YAML config per dataset. See ../prepare_labels_e57.py for the entry point.
"""

from . import lasio, register, c2c, domain, qa  # noqa: F401

__all__ = ["lasio", "register", "c2c", "domain", "qa"]
