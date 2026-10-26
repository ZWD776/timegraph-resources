"""Resolve optional public exports without loading external backends at startup."""
from importlib import import_module


def resolve_optional_export(name, exports, namespace):
    if name not in exports:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    module, _, _ = exports[name]
    group = {key: spec for key, spec in exports.items() if spec[0] == module}
    try:
        loaded = import_module(module, namespace['__package__'])
        values = {key: getattr(loaded, spec[1]) for key, spec in group.items()}
    except ModuleNotFoundError as exc:
        # Preserve the existing optional API: missing types are None, while
        # calls raise the original missing-dependency error when actually used.
        import_error = exc
        def unavailable(*args, **kwargs):
            raise import_error

        values = {key: unavailable if spec[2] else None for key, spec in group.items()}
    namespace.update(values)
    return namespace[name]
