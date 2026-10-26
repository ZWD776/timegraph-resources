"""Resolve compatibility imports to one canonical implementation module."""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys


class _AliasLoader(Loader):
    def __init__(self, alias, target):
        self.alias, self.target = alias, target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Populate the alias after the placeholder was initialized, so the
        # canonical module keeps its own __spec__, __name__ and package identity.
        sys.modules[self.alias] = import_module(self.target)


class _AliasFinder(MetaPathFinder):
    def __init__(self, aliases):
        self.aliases = sorted(aliases.items(), key=lambda pair: len(pair[0]), reverse=True)

    def find_spec(self, fullname, path=None, target=None):
        for source, destination in self.aliases:
            if fullname == source or fullname.startswith(source + '.'):
                selected = destination + fullname[len(source):]
                canonical = import_module(selected)
                return spec_from_loader(fullname, _AliasLoader(fullname, selected),
                                        is_package=hasattr(canonical, '__path__'))
        return None


def install_aliases(aliases):
    sys.meta_path.insert(0, _AliasFinder(aliases))
