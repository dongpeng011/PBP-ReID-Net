# encoding: utf-8
"""A very small, dependency-free replacement for ``yacs.config.CfgNode``.

The project only needs a fraction of the yacs API: hierarchical attribute
access, ``merge_from_file`` (YAML), ``merge_from_list`` (command line
overrides) and ``freeze``.  When ``yacs`` is installed it is used instead (see
``config/__init__.py``); this module is the fallback that keeps the package
runnable on machines where yacs is not available.

Type handling follows yacs closely enough for this project:

* a value read from YAML or from the command line is first decoded with
  ``ast.literal_eval`` when it is a string, so ``"0.0004"``, ``"True"``,
  ``"[16, 16]"`` and ``"('0')"`` all behave as expected;
* the decoded value is then coerced to the type of the existing default
  (``int`` -> ``float`` is allowed, ``list`` <-> ``tuple`` is allowed).
"""

import ast
import copy
import os

import yaml

__all__ = ["CfgNode"]


def _decode(value):
    """Decode a string produced by the command line with literal_eval."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _coerce(new_value, default_value, full_key):
    """Coerce ``new_value`` to the type of ``default_value`` (yacs semantics)."""
    if default_value is None:
        return new_value

    # container defaults: keep the container family of the default
    if isinstance(default_value, (list, tuple)):
        if isinstance(new_value, (list, tuple)):
            items = [_coerce(v, default_value[0] if default_value else None, full_key)
                     for v in new_value]
            return tuple(items) if isinstance(default_value, tuple) else items
        if isinstance(new_value, str):
            # a single string is accepted for a list-of-strings default
            return [new_value]
        raise ValueError("{}: expected a list/tuple, got {}".format(full_key, type(new_value)))

    if isinstance(default_value, bool):
        if isinstance(new_value, bool):
            return new_value
        if isinstance(new_value, str):
            lowered = new_value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
        raise ValueError("{}: expected a bool, got {!r}".format(full_key, new_value))

    if isinstance(default_value, int) and not isinstance(default_value, bool):
        if isinstance(new_value, bool):
            return int(new_value)
        if isinstance(new_value, int):
            return new_value
        if isinstance(new_value, float) and float(new_value).is_integer():
            return int(new_value)
        if isinstance(new_value, str) and new_value.strip().lstrip("-").isdigit():
            return int(new_value)
        raise ValueError("{}: expected an int, got {!r}".format(full_key, new_value))

    if isinstance(default_value, float):
        if isinstance(new_value, (int, float)) and not isinstance(new_value, bool):
            return float(new_value)
        if isinstance(new_value, str):
            try:
                return float(new_value)
            except ValueError:
                pass
        raise ValueError("{}: expected a float, got {!r}".format(full_key, new_value))

    if isinstance(default_value, str):
        return new_value if isinstance(new_value, str) else str(new_value)

    return new_value


class CfgNode(object):
    """Hierarchical, attribute-accessible configuration node."""

    def __init__(self, init_dict=None, key_list=None):
        object.__setattr__(self, "_dict", {})
        object.__setattr__(self, "_frozen", False)
        if init_dict:
            for key, value in init_dict.items():
                self._set(key, value)

    # ---------------------------------------------------------------- basics
    def _set(self, key, value):
        if isinstance(value, dict) and not isinstance(value, CfgNode):
            value = CfgNode(value)
        self._dict[key] = value

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(
                "Configuration key '{}' is not defined".format(name))

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                "Configuration is frozen; '{}' cannot be modified".format(name))
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._set(name, value)

    def __contains__(self, name):
        return name in self._dict

    def __iter__(self):
        return iter(self._dict)

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def get(self, name, default=None):
        return self._dict.get(name, default)

    # ------------------------------------------------------------- freezing
    def freeze(self):
        for value in self._dict.values():
            if isinstance(value, CfgNode):
                value.freeze()
        object.__setattr__(self, "_frozen", True)

    def defrost(self):
        for value in self._dict.values():
            if isinstance(value, CfgNode):
                value.defrost()
        object.__setattr__(self, "_frozen", False)

    def is_frozen(self):
        return self._frozen

    # -------------------------------------------------------------- merging
    def merge_from_file(self, cfg_filename):
        with open(cfg_filename, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            raise ValueError("The config file {} is empty".format(cfg_filename))
        self.merge_from_dict(loaded)

    def merge_from_dict(self, cfg_dict, prefix=""):
        self.defrost()
        for key, value in cfg_dict.items():
            full_key = prefix + key
            if key not in self._dict:
                # new keys are accepted (yacs raises here; relaxing is friendlier
                # for optional PBP-ReID switches added in the yaml files)
                self._set(key, value)
                continue
            current = self._dict[key]
            value = _decode(value)
            if isinstance(current, CfgNode) and isinstance(value, dict):
                current.merge_from_dict(value, prefix=full_key + ".")
            else:
                coerced = _coerce(value, None if isinstance(current, CfgNode) else current, full_key)
                self._set(key, coerced)
        return self

    def merge_from_list(self, cfg_list):
        if len(cfg_list) % 2:
            raise ValueError("Override list has odd length: {}; it must be a list of "
                             "pairs of key and value".format(cfg_list))
        self.defrost()
        for index in range(0, len(cfg_list), 2):
            key, value = cfg_list[index], _decode(cfg_list[index + 1])
            self._merge_key_value(key, value)
        return self

    def _merge_key_value(self, key, value):
        keys = key.split(".")
        node = self
        for name in keys[:-1]:
            if name not in node._dict:
                node._set(name, CfgNode())
            node = node._dict[name]
            if not isinstance(node, CfgNode):
                raise ValueError("Cannot override '{}': '{}' is not a section".format(key, name))
        leaf = keys[-1]
        if leaf not in node._dict:
            node._set(leaf, value)
            return
        current = node._dict[leaf]
        if isinstance(current, CfgNode):
            raise ValueError("Cannot assign a value to the section '{}'".format(key))
        node._set(leaf, _coerce(value, current, key))

    # ------------------------------------------------------------- utilities
    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self):
        out = {}
        for key, value in self._dict.items():
            out[key] = value.to_dict() if isinstance(value, CfgNode) else value
        return out

    def __str__(self):
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=True)

    def __repr__(self):
        return self.__str__()
