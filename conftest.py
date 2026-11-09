# encoding: utf-8
"""Make ``tests/`` importable as ``model`` / ``config`` without installing the package.

``pytest`` inserts the root directory of the repository on ``sys.path`` when a
``conftest.py`` is present there, which is what the tests rely on.
"""
