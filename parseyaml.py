# Copyright Axelera AI, 2023
# TODO: Additional YAML checks

from collections.abc import MutableMapping
import hashlib
import json
import os
import re
import sys
import traceback
from typing import Dict

import yaml

__VAR_DICT = {}


def required_key(key):
    return f"Missing required key: {key}"


def required_value(key):
    return f"Missing value for key: {key}"


def check_key(key, envs, parent=""):
    keystr = f"{parent}:{key}" if parent else key
    if not key in envs:
        print(required_key(keystr))
        return False
    elif not envs[key]:
        print(required_value(keystr))
        return False
    return True


def replace_vars(string: str, val_dict: Dict[str, str], permissive: bool) -> str:
    """Replace any variable defined in the vars section

    Args:
        string (str): The string to replace the variables in
        val_dict (dict[str, str]): The dictionary of variables
        permissive (bool): If true, ignore variables that are not defined

    Returns:
        str -- The string with the variables replaced

    Throws:
        RuntimeError -- If the variable is not defined
    """
    pattern = r'\$\{[^}]*\}'
    instances = []
    while True:
        instances = re.findall(pattern, string)
        if not instances:
            break
        variables = {instance[2:-1]: False for instance in instances}
        for var in variables.keys():
            if var not in val_dict:
                if permissive:
                    continue
                raise RuntimeError(f"Variable {var} is not defined")
            variables[var] = True
            string = string.replace("${" + var + "}", val_dict[var])
        # If we are permissive, we can stop if not all variables are defined
        if permissive and not all(variables.values()):
            break
    return string


def check_vars(key, envs):
    """Check if we have defined any configuration variables"""
    # the vars key is optional and if not defined it can be omitted
    if "vars" not in envs:
        return True
    global __VAR_DICT
    __VAR_DICT = envs['vars'].copy()
    # validate that the variables is a Dict[str, str]
    for k, v in __VAR_DICT.items():
        if not isinstance(k, str):
            raise RuntimeError(f"Variable {k} is not a string")
        if not isinstance(v, str):
            raise RuntimeError(f"Value {v} is not a string")
    # Overwrite the values of the variables from the environment if they exist
    for k, _ in __VAR_DICT.items():
        if k in os.environ:
            __VAR_DICT[k] = os.environ[k]
    # Substitute any variables that reference other variables
    for k, v in __VAR_DICT.items():
        # Non permissive variable substitution: Do not allow any undefined variables in
        # the vars section. Raise an error if we reference a variable that is not defined
        # in the definition of another variable. We expect that we can
        # fully expand the values of the defined variable at parsing time.
        res = replace_vars(v, __VAR_DICT, permissive=False)
        if res:
            __VAR_DICT[k] = res
    envs['vars'] = __VAR_DICT
    return True


def check_os(envs):
    ok = check_key('os', envs)
    if check_key('name', envs['os'], 'os'):
        if not envs['os']['name'] == "Ubuntu":
            print("os:name Installer supports Ubuntu only")
            ok = False
    else:
        ok = False
    ok = ok and check_key('version', envs['os'], 'os')
    return ok


def check_system_libs(key, libs):
    # system libs are always optional
    if key in libs:
        if libs[key]:
            return True
        else:
            print(required_value(key))
            return False
    else:
        return True


def check_penv(envs):
    # penv is optional
    ok = True
    if not 'penv' in envs:
        return True
    if not check_key('penv', envs):
        return False
    envs = envs['penv']
    if check_key('python', envs):
        if not str(envs['python']).startswith("3."):
            print("penv:python: Installer supports python3 only")
            ok = False
    else:
        ok = False
        ok = ok and check_key('pip', envs)
    # TODO expand these
    ok = ok and check_key('repositories', envs)

    if 'libs' not in envs:
        ok = ok and check_key('common', envs)
        ok = ok and check_key('development', envs)
        ok = ok and check_key('runtime', envs)
    else:
        ok = ok and check_key('libs', envs)
    
    ok = ok and check_key('cuda', envs)
    ok = ok and check_key('requirements', envs)
    ok = ok and check_key('pyenv', envs)
    ok = ok and check_key('python_src', envs)
    ok = ok and check_key('python_src_dependencies', envs)
    ok = ok and check_key('pyenv_dependencies', envs)
    return ok


def validate(envs):
    # name and os are required
    # dependencies, penv, docker, runtime and driver and repositories, symlinks and next are optional
    ok = True
    ok = ok and check_key('name', envs)
    ok = ok and check_vars('vars', envs)
    ok = ok and check_os(envs)
    ok = ok and check_system_libs('dependencies', envs)
    ok = ok and check_penv(envs)
    # TODO expand these
    ok = ok and check_key('docker', envs)
    ok = ok and check_key('runtime', envs)
    ok = ok and check_key('driver', envs)
    # repositories
    # symlinks
    # next
    return ok


def limit_for_hash(env, hash_type):
    # Remove all settings that don't affect
    # docker contents
    is_docker_hash = hash_type == "docker_hash"
    if 'penv' in env:
        if 'repositories' in env['penv']:
            del env['penv']['repositories']
        if 'libs' in env['penv']:  # requirements
            del env['penv']['libs']
        if 'cuda' in env['penv']:  # requirements
            del env['penv']['cuda']
        if 'pyenv' in env['penv']:
            del env['penv']['pyenv']
        if 'python_src' in env['penv']:
            del env['penv']['python_src']
        if 'pyenv_dependencies' in env['penv']:
            del env['penv']['pyenv_dependencies']
    if 'docker' in env:
        if not is_docker_hash:
            del env['docker']
        else:
            if 'system-dependencies' in env['docker']:
                del env['docker']['system-dependencies']
            if 'system-cuda-dependencies' in env['docker']:
                del env['docker']['system-cuda-dependencies']
    if 'runtime' in env:
        if not is_docker_hash:
            del env['runtime']
        else:
            if 'description' in env['runtime']:
                del env['runtime']['description']
    if 'driver' in env:
        del env['driver']
    if 'repositories' in env:
        del env['repositories']
    if 'media' in env:
        del env['media']
    if 'symlinks' in env:
        del env['symlinks']
    if 'next' in env:
        del env['next']


def flatten(env, parent=False):
    items = []
    for key, value in env.items():
        key = str(parent) + str("_") + str(key) if parent else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, key).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, key).items())
        else:
            items.append((key, value))
    return dict(items)


def usage(name):
    print(f"Usage: python3 {name} file [(docker_hash|env_hash) requirements]")

def create_penv_libs(flat):
    # create penv_libs_ from penv_common_, penv_development_ and penv_runtime_
    base = "penv"
    envs = ['common', 'development', 'runtime']
    regex = re.compile(rf"{base}_(?:{'|'.join(envs)})_\d+_(.*)")

    count = 0
    libs = {}
    for key, value in flat.items():
        if m := regex.match(key):
            libs[f"{base}_libs_{count}_{m.group(1)}"] = value
            count += 1

    flat.update(libs)

def main():
    envs = {}

    assert len(sys.argv) == 2 or len(sys.argv) == 4, usage(sys.argv[0])

    with open(sys.argv[1], "r") as f:
        envs = yaml.safe_load(f)

    if not validate(envs):
        sys.exit(1)

    if len(sys.argv) == 4 and (sys.argv[2] == "docker_hash" or sys.argv[2] == "env_hash"):
        limit_for_hash(envs, sys.argv[2])
        yaml_str = json.dumps(envs, sort_keys=True)
        with open(sys.argv[3], 'r') as file:
            yaml_str += file.read()
        hash_str = hashlib.sha256(yaml_str.encode('utf-8')).hexdigest()
        print(hash_str[:8])

    elif len(sys.argv) == 2:
        flat = flatten(envs)
        create_penv_libs(flat)
        for i in flat.items():
            # Substitute the values of any variables
            key = i[0].replace('-', '_').replace('.', '_')
            # Variable substitution is permissive in the body of the configuration
            # If there are undefined variables then we expect that those are
            # system environment variables that will be defined at runtime.
            value = replace_vars(str(i[1]), __VAR_DICT, permissive=True)
            value=str(value).replace('"', '\\"').replace('$', '\\\\\\$')
            print(f'export AX_{key}="{value}"')
    else:
        usage(sys.argv[0])
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        sys.exit(1)
    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)
    except:
        _, evalue, _ = sys.exc_info()
        print(evalue)
        print(traceback.format_exc())
        sys.exit(1)
