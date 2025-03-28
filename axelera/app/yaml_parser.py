#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Makefile and build system integration

import collections
import difflib
import os
from pathlib import Path
import textwrap
from typing import List

import yaml

_YELLOW = "\x1b[33;20m"
_RESET = "\x1b[0m"


NN_FILE_MSGS = []


def any_parse_msgs():
    global NN_FILE_MSGS
    if NN_FILE_MSGS:
        if len(NN_FILE_MSGS) == 1:
            msg = NN_FILE_MSGS[0]
        else:
            msg = ["Errors/warnings found in multiple model files:"] + NN_FILE_MSGS
            msg = "\n\n".join(msg)
        NN_FILE_MSGS = []
        return msg


def get_element(dict, el):
    val = ""
    if dict and el in dict:
        val = dict[el]
    return val


_EXCLUDE_DIRS = ['training_yamls', 'yolo/cfg']


def get_model_files(dir, exclude_dirs):
    all_exclude_dirs = _EXCLUDE_DIRS + exclude_dirs
    files = []
    for path in Path(dir).rglob('*.yaml'):
        if path.is_file() and not any([d in str(path) for d in all_exclude_dirs]):
            files.append(str(path))
    return files


_AXELERA_FORMAT_KEY = 'axelera-model-format'

_EXPECTED_MODEL_FORMAT = '1.0.0'

_expected_model_keys = {
    None: set(['name', 'description', 'pipeline', 'models', 'datasets']),
    '1.0.0': set([_AXELERA_FORMAT_KEY, 'name', 'description', 'pipeline', 'models', 'datasets']),
}


def model_sanity_check(model, path) -> List[str] | None:
    # check for expected sections
    # TODO: could be an enforced schema instead
    msgs = []
    axelera_model_format = model.get(_AXELERA_FORMAT_KEY)
    if axelera_model_format != _EXPECTED_MODEL_FORMAT:
        msgs += [
            f'{path}: Expected {_AXELERA_FORMAT_KEY}: {_EXPECTED_MODEL_FORMAT} but found {axelera_model_format}'
        ]

    expected_keys = _expected_model_keys.get(_EXPECTED_MODEL_FORMAT)
    model_keys = set(model.keys())
    if not expected_keys.issubset(model_keys):
        if model_keys == _expected_model_keys.get(None):
            msgs += [
                f'{path}: Accepting unversioned model since it contains the expected sections'
            ]
        else:
            msgs += [
                f"Unusable model: '{path}' since it does not contain the expected sections {expected_keys}\n(it has {model_keys})"
            ]
            NN_FILE_MSGS.append("\n".join(msgs))
            return None
    return msgs


def model_from_path(path):
    try:
        with open(path, "r") as y:
            if model := yaml.safe_load(y):
                msgs = model_sanity_check(model, path)
                if msgs is None:
                    return None
                if msgs:
                    NN_FILE_MSGS.append("\n".join(msgs))
                return model
    except yaml.scanner.ScannerError as e:
        NN_FILE_MSGS.append(str(e))
        return None


def check_unique(models):
    seen_name = {}
    seen_description = {}
    duplicated = set()
    for name, path, model, _, _ in models:
        if name in seen_name:
            NN_FILE_MSGS.append(
                f"Unusable model: '{name}' since it is duplicated in {seen_name[name][0]} and {path}"
            )
            duplicated.add(name)
        else:
            # warn if models have the same description, but not if one appears to be an
            # ax- dev equivalent of the other
            description = get_element(model, "description")
            seen_name[name] = path, description
            if description in seen_description:
                that = seen_description[description]
                that = that[0], str(that[1])
                this = name, str(path)
                ax_prefix = "ax-"
                if this[0].startswith(ax_prefix) and not that[0].startswith(ax_prefix):
                    this, that = that, this
                if ax_prefix + this[0] != that[0] or not that[1].startswith("ax_models/dev/"):
                    NN_FILE_MSGS.append(
                        f"info: models '{this[0]}' and '{that[0]}' have the same description:\n- {this[1]}\n- {that[1]}\nDescription: {description}"
                    )
                    continue

            seen_description[description] = name, path

    if duplicated:
        # remove any which are duplicated, to force fail if attempting to use
        models = [m for m in models if m[0] not in duplicated]

    return models


def get_models(dir, prefix, build_root="build", exclude_dirs=[]):
    files = get_model_files(dir, exclude_dirs=exclude_dirs)
    models = [(path, model_from_path(path), prefix, build_root) for path in files]
    models = [m for m in models if m[1] is not None]
    models = [(get_element(m[1], 'name'), *m) for m in models]
    models = sorted(models, key=lambda x: x[0])
    models = check_unique(models)
    return models


_print_width = 120
_name_min_width = 25
# empirical values that look good between 80 and 120 columns
_name_width = max(_name_min_width, _print_width // 2 - _name_min_width)
_line_width = _print_width - _name_width - 4


def print_header():
    print('+' + '-' * (_print_width - 2))


def print_separator():
    print('|' + '=' * (_print_width - 2))


def print_line(line=""):
    print(f"| {line:<{_print_width - 4}} |")


def print_newline():
    print_line("")


def print_col2(name, desc):
    lines = textwrap.wrap(desc, _line_width, break_long_words=True) if desc else []
    if len(name) >= _name_width:
        print(f"| {name:<{_print_width - 4}} |")
        name = ""
    for line in lines:
        print(f"| {name:<{_name_width}}{line:<{_line_width}} |")
        name = ""


def print_model_help(models, title):
    first = True
    for name, _, model, prefix, _ in models:
        desc = get_element(model, 'description')
        if first:
            print_line(title)
        if name:
            print_col2(f"  {prefix}{name}", desc)
        first = False


def _space(text):
    return text.replace(" ", "_SP_").replace("\t", "_SP_")


def gen_model_envs(nn):
    model_collection = get_model_details()
    models = model_collection.all_models()

    if msgs := any_parse_msgs():
        for msg in msgs.split("\n"):
            print(_space(f"$(warning {_YELLOW}WARNING: {_RESET}{msg})"))

    for name, path, model, prefix, build in models:
        if f'{prefix}{name}' == nn:
            desc = get_element(model, 'description')
            print(_space(f"NN_FILE:={path}"))
            print(_space(f"NN_BUILD:={build}{os.sep}{name}"))
            print(_space(f"NN_DESC:={desc}"))
            print(_space(f"NN_DEPS:={path}"))
            root = Path(path).parent
            for model_name, model_yaml in model.get('models', {}).items():
                print(_space(f"NN_MODELS+={model_name}.o"))
                print(_space(f"NN_DEPS+={root/model_yaml['class_path']}"))
            print(_space(f"NN_SET:=1"))
            break
    else:
        print("NN_SET:=0")


class ModelCollection(
    collections.namedtuple(
        'ModelCollection', ['local', 'zoo', 'cards', 'customers', 'reference', 'tutorial']
    )
):
    def all_models(self):
        """Returns a flat list of all models combined."""
        return sum(self._asdict().values(), [])

    def all_network_names(self):
        """Returns a set of all network names in format 'prefix+name'."""
        return {nn[3] + nn[0] for model_list in self._asdict().values() for nn in model_list}


def get_model_details(
    model_cards_only: bool = False,
    check_customer_models: bool = False,
    apply_framework_dir: bool = True,
) -> ModelCollection:
    prefix = os.path.expandvars("$AXELERA_FRAMEWORK/") if apply_framework_dir else ""
    model_cards = get_models(f"{prefix}ax_models/model_cards", "mc-")
    model_customers = get_models(f"{prefix}customers", "c-")
    if model_cards_only:  # Return only model-cards for CI tests
        customer_cards = []
        if check_customer_models:
            customer_cards = [card for card in model_customers if 'internal-model-card' in card[2]]
        return ModelCollection(
            [], [], model_cards, customer_cards, [], []
        )  # TODO: include model_customers in CI

    model_reference = get_models("ax_models/reference", "")
    model_tutorial = get_models("ax_models/tutorials", "")
    model_local = get_models(
        "ax_models",
        "",
        exclude_dirs=["zoo", "model_cards", "customers", "reference", "tutorials"],
    )
    model_zoo = get_models(f"{prefix}ax_models/zoo", "")
    return ModelCollection(
        local=model_local,
        zoo=model_zoo,
        cards=model_cards,
        customers=model_customers,
        reference=model_reference,
        tutorial=model_tutorial,
    )


def gen_model_help():
    model_collection = get_model_details()

    print_header()
    print_line("Axelera AI - Voyager SDK")
    print_header()
    print_newline()
    print_line("Usage:")
    print_line("  make [ NN=<network> ] [ target ]")
    print_newline()
    print_separator()
    print_line("TARGETS")
    print_col2("  clean NN=<network>", "Clean network's build folder")
    print_col2("  info NN=<network>", "Print information about network")
    print_col2(
        "  operators", "Build pre and post processing operators for gstreamer and AxInferenceNet"
    )
    print_col2("  trackers", "Build tracking support library")
    print_col2("  clean-libs, clobber-libs", "Clean/clobber operators and trackers")
    print_col2("  update", "Re-parse network yamls for make")
    print_col2("  help", "Print this help information")
    print_separator()
    print_model_help(model_collection.local, "MODELS")
    print_model_help(model_collection.zoo, "ZOO")
    print_model_help(model_collection.cards, "INTERNAL MODEL CARDS")
    print_model_help(model_collection.tutorial, "TUTORIALS")
    print_model_help(model_collection.reference, "REFERENCE APPLICATION PIPELINES")
    print_model_help(model_collection.customers, "CUSTOMER MODELS [Confidential]")

    print_header()


NetworkYamlBase = collections.namedtuple(
    'NetworkYamlBase', ['name', 'yaml_name', 'yaml_path', 'cascaded']
)


class NetworkYamlInfo:
    def __init__(self):
        self.info = {}

    def add_info(self, name: str, yaml_path: str, model: dict, prefix: str, build: str) -> None:
        yaml_name = f'{prefix}{name}'
        if not (models := model.get('models', {})):
            raise ValueError(f"{yaml_path} has no models section!")
        cascaded = len(models) > 1
        info = NetworkYamlBase(name, yaml_name, yaml_path, cascaded)

        self.info[yaml_name] = info
        self.info[yaml_path] = info

    def get_info(self, key: str):
        try:
            resolved_key = str(Path(key).resolve())
        except Exception:
            resolved_key = key

        if (info := self.info.get(key) or self.info.get(resolved_key)) is not None:
            return info

        suffix = '.yaml' if key.endswith('.yaml') else ''
        all_keys = (
            [k for k in self.info if k.endswith(suffix)] if suffix else self.get_all_yaml_names()
        )
        likely = difflib.get_close_matches(key, all_keys, n=5)

        error_message = f"Invalid network '{key}', "
        if likely:
            suggestions = ', '.join(likely)
            error_message += f"did you mean {'one of: ' if len(likely) > 1 else ''}{suggestions}?"
        else:
            error_message += (
                "no close match found. Please `make help` to see all available models."
            )
        raise KeyError(error_message)

    def get_all_info(self):
        # Since both names and paths point to the same objects, deduplicate them
        unique_infos = list({id(info): info for info in self.info.values()}.values())
        return unique_infos

    def get_all_yaml_names(self):
        # Filter out paths to only return names
        return [key for key in self.info if not key.endswith('.yaml')]


def _get_network_yaml_info(models):
    network_yaml_info = NetworkYamlInfo()
    for name, path, model, prefix, build in models:
        network_yaml_info.add_info(name, path, model, prefix, build)
    return network_yaml_info


def get_network_yaml_info(
    model_cards_only: bool = False,
    check_customer_models: bool = False,
    apply_framework_dir: bool = True,
    include_collections: List[str] | None = None,
):
    """Get network YAML information for specified model collections.

    Args:
        model_cards_only: If True, only include model cards
        check_customer_models: If True, include customer model cards when model_cards_only is True
        apply_framework_dir: If True, prepend AXELERA_FRAMEWORK to paths
        include_collections: List of collections to include. If None, includes all.
                           Valid values: ['local', 'zoo', 'cards', 'customers', 'reference', 'tutorial']
    """
    model_collection = get_model_details(
        model_cards_only, check_customer_models, apply_framework_dir
    )

    if include_collections is not None:
        # Filter to only requested collections
        all_models = []
        collection_dict = model_collection._asdict()
        for collection in include_collections:
            if collection in collection_dict:
                all_models.extend(collection_dict[collection])
    else:
        all_models = model_collection.all_models()

    if msgs := any_parse_msgs():
        print(f"{_YELLOW}WARNING: {_RESET}{msgs}")
    return _get_network_yaml_info(all_models)
