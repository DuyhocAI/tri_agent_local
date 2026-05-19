"""Resolve a dotted-path string to a callable at import time."""

import importlib


def resolve_handler(dotted_path: str):
    """Import and return the callable at 'package.module.function_name'.

    Example: resolve_handler("skills.handlers.read_file")
    """
    module_path, _, func_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid handler path (no module): {dotted_path!r}")
    module = importlib.import_module(module_path)
    if not hasattr(module, func_name):
        raise AttributeError(f"{module_path!r} has no attribute {func_name!r}")
    return getattr(module, func_name)
