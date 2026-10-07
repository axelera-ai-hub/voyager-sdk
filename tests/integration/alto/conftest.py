# Copyright Axelera AI, 2026
"""Keep pytest out of the mirrored ALTO package.

Nothing here is a test of this repository's code: the tree exists so the imported ALTO
files resolve their own absolute import names. One file carries an upstream ``test_``
name (see ``__init__.py``) and would otherwise be collected, and importing it pulls in the
compiler, which two of the tox environments do not install.
"""

collect_ignore_glob = ["*"]
