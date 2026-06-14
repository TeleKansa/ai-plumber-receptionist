"""Industry-agnostic prompt/session engine.

Nothing vertical-specific may live in this package. Vertical behavior comes
entirely from a config in verticals/<name>.json, loaded via
core.vertical.load_vertical and rendered by core.engine. Adding a new industry
should require only a new vertical config file — zero changes here. This
invariant is enforced by an automated guard test.
"""
