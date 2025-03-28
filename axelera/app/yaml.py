# Copyright Axelera AI, 2023
# Define types for use in dictionaires of YAML which contain line numbers

import yaml

from . import logging_utils

LOG = logging_utils.getLogger(__name__)


class LINE_LOADER(yaml.SafeLoader):
    pass


class AxYAMLError(yaml.YAMLError):
    def __init__(self, message, key_or_path):
        if isinstance(key_or_path, dict):
            super().__init__(_yaml_error_parent(key_or_path, message))
        elif hasattr(key_or_path, 'line'):
            super().__init__(_yaml_error(key_or_path, message))
        elif key_or_path is not None:
            super().__init__(message + '\n' + key_or_path)
        else:
            super().__init__(message)


class MapYAMLtoFunction:
    """Map YAML attributes to positions in function calls including args and kwargs,
    ignoring any attributes not in the function call."""

    param_supported = []
    param_required = []
    param_defaults = {}
    yaml_named_args = {}
    yaml_attribs = {}

    def __init__(self, supported, required, defaults, named_args, attribs):
        self.param_supported = supported
        self.param_required = required
        self.param_defaults = defaults
        self.yaml_named_args = named_args
        self.yaml_attribs = attribs

    def get_arglist(self, args):
        # Return an ordered list of initialized args
        arg_list = []
        if not isinstance(args, list):
            args = [args]
        for arg in args:
            if not arg in self.yaml_named_args:
                raise RuntimeError(f"Attribute '{arg}' is not a named argument")
            arg_list.append(self.get_arg(arg))
        return arg_list

    def get_arg(self, arg):
        # Get argument value
        if arg in self.yaml_attribs:
            return self.yaml_attribs[arg]
        elif arg in self.param_defaults:
            return self.param_defaults[arg]
        else:
            raise AxYAMLError(f"Missing attribute '{arg}'", self.yaml_attribs)

    def get_kwargs(self):
        # Return a dictionary of initialized kwargs
        args = {}
        for k, v in self.yaml_attribs.items():
            if k in self.param_supported and not k in self.yaml_named_args:
                args[k] = v
        for item in self.param_required:
            if not item in self.yaml_named_args and not item in self.yaml_attribs:
                raise AxYAMLError(f"Missing attribute '{item}'", self.yaml_attribs)
        return args


class yamlBool(int):
    # Unable to extend bool type so use int
    # instead with 1 and 0
    def __new__(cls, value=False, line=None):
        if (isinstance(value, str) and value.lower() == "true") or value == 1:
            i = int.__new__(cls, 1)
        else:
            i = int.__new__(cls, 0)
        i._line = line
        i._parent = None
        return i

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        raise Exception("line number is read-only")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value


class yamlInt(int):
    def __new__(cls, value=0, line=None):
        if isinstance(value, str) and value.startswith(("0x", "0X")):
            i = int.__new__(cls, value, 0)
        else:
            i = int.__new__(cls, value)
        i._line = line
        i._parent = None
        return i

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        raise Exception("Line number is read-only")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value


class yamlFloat(float):
    def __new__(cls, value=0, line=None):
        f = float.__new__(cls, value)
        f._line = line
        f._parent = None
        return f

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        raise Exception("line number is read-only")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value


class yamlStr(str):
    def __new__(cls, value="", line=None):
        s = str.__new__(cls, value)
        s._line = line
        s._parent = None
        return s

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        raise Exception("line number is read-only")

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value


def construct_yaml_bool(loader, node):
    value = loader.construct_scalar(node)
    return yamlBool(value, node)


def construct_yaml_int(loader, node):
    value = loader.construct_scalar(node)
    return yamlInt(value, node.start_mark)


def yaml_int_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:int', repr(data))


def construct_yaml_float(loader, node):
    value = loader.construct_scalar(node)
    return yamlFloat(value, node.start_mark)


def construct_yaml_str(loader, node):
    value = loader.construct_scalar(node)
    return yamlStr(value, node.start_mark)


LINE_LOADER.add_constructor('tag:yaml.org,2002:bool', construct_yaml_bool)
LINE_LOADER.add_constructor('tag:yaml.org,2002:int', construct_yaml_int)
yaml.add_representer(yamlInt, yaml_int_representer)
LINE_LOADER.add_constructor('tag:yaml.org,2002:float', construct_yaml_float)
LINE_LOADER.add_constructor('tag:yaml.org,2002:str', construct_yaml_str)


def attribute(d, option, allow_none=False):
    return require_attribute(d, option, allow_none)


def require_attribute(d, options, allow_none=False):
    name = ""
    if isinstance(options, tuple):
        for option in options:
            if option in d:
                if name:
                    options = ', '.join(str(s) for s in options)
                    raise AxYAMLError(f"Only one of following attributes: {options}", d)
                else:
                    name = option
        if not name:
            options = ', '.join(str(s) for s in options)
            raise AxYAMLError(f"Missing one of following attributes: {options}", d)
    else:
        name = options
    if d is None or name not in d:
        raise AxYAMLError(f"Missing attribute '{name}'", d)
    value = d[name]
    if value is None and not allow_none:
        raise AxYAMLError(f"Attribute is unexpectedly None", key(d, name))
    return value


def _yaml_error(attribute, msg):
    if hasattr(attribute, 'line'):
        return str(attribute) + ': ' + msg + '\n' + str(attribute.line)
    return msg


def _yaml_error_parent(d, msg):
    assert isinstance(d, dict)
    a_child = tuple(d.items())[0][0]
    if hasattr(a_child, 'parent') and hasattr(a_child.parent, 'line'):
        return a_child.parent + f": {msg}" + '\n' + str(a_child.parent.line)
    return msg


def key(d, name):
    for k, v in d.items():
        if k == name:
            return k
    return None
