from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ree"
DEV_TOOLS = {"pytest", "hypothesis", "ruff", "httpx2"}
TEST_CLIENT_PACKAGES = {"httpx", "httpx2", "httpcore", "httpcore2"}
IMPORT_TO_DISTRIBUTION = {"yaml": "pyyaml"}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _names(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[;\s]", item, maxsplit=1)[0].lower() for item in requirements}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_dev_tools_are_declared_for_both_uv_and_pip() -> None:
    project = _project()
    group = project["dependency-groups"]["dev"]
    extra = project["project"]["optional-dependencies"]["dev"]
    assert sorted(group) == sorted(extra)
    assert _names(group) >= DEV_TOOLS


def test_runtime_dependencies_do_not_include_dev_tools() -> None:
    declared = _names(_project()["project"]["dependencies"])
    assert not declared & (DEV_TOOLS | TEST_CLIENT_PACKAGES)


def test_every_third_party_import_in_the_package_is_a_declared_dependency() -> None:
    declared = _names(_project()["project"]["dependencies"])
    undeclared: dict[str, set[str]] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        for root in _imported_roots(path):
            if root == "ree" or root in sys.stdlib_module_names or root == "__future__":
                continue
            distribution = IMPORT_TO_DISTRIBUTION.get(root, root).lower()
            if distribution not in declared:
                undeclared.setdefault(root, set()).add(path.name)
    assert undeclared == {}


def test_package_code_never_imports_a_test_client() -> None:
    offenders = {}
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = _imported_roots(path) & TEST_CLIENT_PACKAGES
        if "testclient" in text or found:
            offenders[path.name] = sorted(found) or ["testclient"]
    assert offenders == {}
