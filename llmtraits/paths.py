"""Path resolution helper.

The full project this was extracted from supported moving large model/delta
directories between several physical drives over time, with a small
rewrite table mapping old locations to new ones. That table was entirely
specific to one machine's drive layout, so it has been removed here.

``relocated_path``/``relocated_str`` are kept as no-op passthroughs so the
rest of the package (which calls them on every stored path, in case a
future setup wants to reintroduce relocation) keeps working unchanged. If
you need the same "a directory moved to another drive" behavior, add your
own rewrite rules in ``relocated_path`` below.
"""

from __future__ import annotations

import os
from pathlib import Path


def relocated_path(path: str | os.PathLike[str]) -> Path:
    """Return *path* unchanged. Override this if your storage layout moves."""

    return Path(path)


def relocated_str(path: str | os.PathLike[str]) -> str:
    return str(relocated_path(path))
