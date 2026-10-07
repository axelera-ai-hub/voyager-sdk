# Copyright Axelera AI, 2026
"""Mirror of the ALTO integration-test package path, so the imported tree runs unedited.

The LLM toolkit in this repository was extracted from
``axelera_compiler/tests/integration/alto/`` in the compiler checkout, and every file the
ALTO team ports across still imports it by that absolute name. Recreating the name here
costs one shim; editing the import lines instead would cost the porting team a re-applied
delta on every future port, so the name is what moves.

Three pieces make the name resolve:

* ``tests.integration.alto.llm`` is aliased onto :mod:`model_axscript.llm` below.
* ``tests.integration.alto.utils`` beside this file is a vendored subset of the upstream
  harness.
* ``tests.integration.alto.axol.core.test_core_scaled_dot_product_attention`` is a vendored
  subset of the upstream core AXOL SDPA test, kept under its upstream name for the same
  reason.

Nothing in this package is a test of this repository's own code; ``conftest.py`` beside
this file keeps pytest from collecting it.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_ALIAS = "tests.integration.alto.llm"
_TARGET = "model_axscript.llm"


class _ToolkitAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves the mirrored package name onto the real toolkit's module objects.

    Aliasing rather than symlinking or copying keeps exactly one module instance per
    module, so an import-time registry is populated once whichever name reached it first,
    and an ``isinstance`` check across the two names holds.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _ALIAS or fullname.startswith(_ALIAS + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_TARGET + spec.name[len(_ALIAS) :])

    def exec_module(self, module):
        """No-op: the module returned by create_module already ran under its real name."""


sys.meta_path.insert(0, _ToolkitAlias())
