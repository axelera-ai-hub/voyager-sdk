# Copyright Axelera AI, 2024
# Utility functions used by the Voyager SDK
from __future__ import annotations

from contextlib import contextmanager
from enum import Enum, EnumMeta, IntEnum
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from typing import Callable, Optional, Union
import zipfile

import psutil
import requests
from tqdm import tqdm
import yaml

from . import logging_utils
from .environ import env
from .yaml import LINE_LOADER

LOG = logging_utils.getLogger(__name__)

IMAGE_EXTENSIONS = [
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".gif",
    ".png",
    ".eps",
    ".raw",
    ".cr2",
    ".nef",
    ".orf",
    ".sr2",
    ".bmp",
    ".ppm",
    ".heif",
]
VIDEO_EXTENSIONS = [".flv", ".avi", ".mp4", ".3gp", ".mov", ".webm", ".ogg", ".qt", ".avchd"]


def load_yamlfile(path: Union[str, Path]):
    """Load data from YAML file"""
    path = Path(path)
    data_dict = yaml.load(path.read_text('utf-8'), Loader=LINE_LOADER)
    return data_dict


def load_yaml_by_reference(path: Union[str, Path], refs: dict):
    """Load a YAML file with variable and expression substitution.

    Replace $?{{keyword}} with value from refs[keyword].
    Replace ${expr} with str(eval(expr)).

    """
    special_values = {"None": "null", "True": "true", "False": "false"}

    def subst(m):
        key, default = m.group(1, 2)
        try:
            val = str(refs[key])
            return special_values.get(val, val)
        except KeyError:
            if default is not None:
                return default
            LOG.error(f"{path}: Variable {key} missing argument from {refs}")
            return m.group(0)

    def evalexpr(m):
        return str(eval(m.group(1), refs))

    out = []
    path = Path(path)
    stream = path.read_text('utf-8')
    for line in stream.splitlines():
        line = re.sub("^(.*)(?:#.*)$", r"\1", line)
        line = re.sub(r"\$?{{([^:}]+)(?::([^}]+))?}}", subst, line)
        line = re.sub(r"\${([^{}][^}]*)}", evalexpr, line)
        out.append(line)
    return yaml.load("\n".join(out), Loader=yaml.FullLoader)


def load_yaml_ignore_braces(path: Union[str, Path]):
    """
    Load data from YAML file and all occurrences of "{{" and "}}"
    """
    path = Path(path)
    stream = path.read_text('utf-8')
    stream = re.sub(r"\{\{.*?\}\}", "", stream)
    return yaml.load(stream, Loader=yaml.FullLoader)


def visit_dict(obj, visitor):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                visit_dict(v, visitor)
            else:
                newv = visitor(k, v)
                if newv != v:
                    LOG.trace(f"Changing {k} from {v} to {newv}")
                    obj[k] = newv
    elif isinstance(obj, list):
        for v in obj:
            visit_dict(v, visitor)


def _make_path_absolute(base, path):
    """
    Convert a path to an absolute path relative to the base directory.
    Handles environment variables and home directory (~) expansion.
    """
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    if not Path(path).is_absolute():
        return str(Path(base, path).resolve())
    return path


def make_path_absolute(k, v, base, key_suffixes):
    key_alone = [k[1:] for k in key_suffixes]
    if k in key_alone or k.endswith(key_suffixes) and isinstance(v, str):
        v = _make_path_absolute(base, v)
    return v


def make_weight_path_absolute(k, v, base, keys, model_name):
    if k in keys and isinstance(v, str):
        v = _make_path_absolute(base / model_name, v.removeprefix("weights/"))
    return v


def make_paths_in_dict_absolute(base, obj, key_suffixes=("_path", "_dir", "_root")):
    """
    Convert all path values associated with keys ending in specific
    suffixes to absolute paths relative to the base directory
    """

    def visitor(k, v):
        return make_path_absolute(k, v, base, key_suffixes)

    visit_dict(obj, visitor)


def make_weight_paths_in_dict_absolute(base, obj, keys=("weight_path")):
    """
    Convert all weight path values to absolute paths relative to the base directory
    """

    models = obj.get("models")
    if not models:
        return

    for model in models.items():
        visit_dict(
            model[1], lambda k, v: make_weight_path_absolute(k, v, base, keys, str(model[0]))
        )


def get_classes_from_module(module_name):
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    classes = [
        member[0]
        for member in inspect.getmembers(module, inspect.isclass)
        if member[1].__module__ == module.__name__
    ]
    return classes


def import_module_from_file(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"{module_path}: Failed to import module {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_class_from_file(class_name, module_path):
    """Import a class type definition from a file"""
    module_name = Path(module_path).stem.replace('-', '_')
    try:
        module = import_module_from_file(module_name, module_path)
    except Exception as e:
        raise e.__class__(
            f"{module_path}: Failed to import module {module_name} : {e}"
        ).with_traceback(e.__traceback__)
    try:
        return getattr(module, class_name)
    except Exception:
        raise ImportError(
            f"{module_path}: {module_name} does not contain the class {class_name}"
        ) from None


def import_from_module(module, item):
    return importlib.import_module(module).__getattribute__(item)


def is_method_overridden(instance, method_name):
    """
    Determines if a base class method has been overridden in a derived instance.

    Args:
        instance: The instance of the derived class.
        method_name: The name of the method to check.

    Returns:
        bool: True if the method has been overridden, False otherwise.
    """
    method = getattr(instance.__class__, method_name, None)
    base_methods = [getattr(C, method_name, None) for C in instance.__class__.__mro__[1:]]
    return method is not None and any(bm is not None and method != bm for bm in base_methods)


def get_media_type(file):
    ext = Path(file).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    raise ValueError(f"Not supported format for {file}")


def find_values_in_dict(key, dictionary):
    """
    Recursively search nested dictionary for all values of a given key.
    """
    values = []
    for k, v in dictionary.items():
        if k == key:
            values.append(v)
        elif isinstance(v, dict):
            values.extend(find_values_in_dict(key, v))
    return values


def _check_tar_gz(path: Path):
    return path.suffix == '.gz' and path.stem.endswith('.tar')


def download_and_extract_asset(
    url: str,
    path: Path,
    md5: str = "",
    dest_path: Path = None,
    delete_archive: bool = True,
    drop_dirs: int = 0,
):
    # Dowload file and, if archive, extract
    download(url, path, md5)
    if path.suffix in {'.zip', '.tar', '.tgz'} or _check_tar_gz(path):
        extract(path, drop_dirs=drop_dirs, dest=dest_path)
        if delete_archive:
            path.unlink()
        else:
            LOG.debug(f"{path} uncompressed. You may now safely delete this file")


def download(url: str, path: Path, checksum=""):
    """Download file from URL and validate checksum"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (not checksum or md5_validates(path, checksum)):
        return False  # file exists

    if url.startswith('s3://') or url.startswith('S3://'):
        parts = url[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError("Invalid S3 path format. Path should have a bucket and a key.")
        local_file = download_from_internal_s3(parts[0], parts[1], path.parent)
        if checksum and not md5_validates(local_file, checksum):
            raise RuntimeError(f"Downloaded file {path} failed CRC check")
        shutil.move(local_file, path)
        return True

    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        with open(path, 'wb') as f:
            progress = tqdm(
                total=int(request.headers['Content-Length']),
                desc="Download %s" % path,
                unit='B',
                unit_divisor=1024,
                unit_scale=True,
                leave=False,
            )
            for chunk in request.iter_content(chunk_size=8192):
                if chunk:  # ignore keep-alive
                    f.write(chunk)
                    progress.update(len(chunk))
            progress.close()
    if checksum and not md5_validates(path, checksum):
        raise RuntimeError(f"Downloaded file {path} failed CRC check")
    return True


def download_from_internal_s3(s3_bucket_name: str, s3_filepath: str, cache_path: Path) -> str:
    import boto3

    cache_path.mkdir(parents=True, exist_ok=True)
    local_filepath = cache_path.joinpath(Path(s3_filepath).name)

    if not local_filepath.exists():
        LOG.info(f"Downloading {s3_filepath} from s3 bucket {s3_bucket_name}")
        s3_resource = boto3.resource("s3")
        s3_object = s3_resource.Object(s3_bucket_name, s3_filepath)
        with spinner() as progress:
            progress.title(f"Downloading {local_filepath.name}")
            s3_object.download_file(str(local_filepath))

    return local_filepath


def dir_needed(path):
    """Check if path needs to be created. If path exists but is not a dir, then raise exception"""
    if path.exists():
        if path.is_dir():
            return False
        else:
            raise RuntimeError(f"{path}: Not a directory")
    return True


def generate_md5(path: str) -> str:
    hash_md5 = hashlib.md5()
    nbytes = path.stat().st_size
    disable = True if nbytes < 1000000000 else False
    with open(path, 'br') as f:
        for chunk in tqdm(
            iter(lambda: f.read(4096), b""),
            total=int(os.fstat(f.fileno()).st_size / 4096),
            desc="Authenticate %s" % path,
            unit='chunk',
            unit_scale=True,
            unit_divisor=4096,
            disable=disable,
            leave=False,
        ):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def md5_validates(path, md5):
    if not md5:
        return False
    return generate_md5(path) == md5


def extract(path: Path, drop_dirs: int = 0, dest: Path = None):
    if not path.exists():
        raise RuntimeError(f"{path}: File does not exist")
    elif path.suffix in {'.tar', '.tgz'} or _check_tar_gz(path):
        extract_tar(path, drop_dirs, dest)
    elif path.suffix == '.zip':
        extract_zip(path, drop_dirs, dest)
    else:
        raise RuntimeError(f"{path}: Unsupported archive format/extension")


def extract_tar(path, drop_dirs: int = 0, dest: Path = None):
    mode = 'r:' if path.suffix == '.tar' else 'r:gz'
    with tarfile.open(path, mode) as tar:
        for member in tqdm(tar.getmembers(), desc=f'Uncompress {path}', unit='files', leave=False):
            if drop_dirs > 0:
                parts = Path(member.name).parts[drop_dirs:]
                if not parts:
                    continue
                member.name = str(Path(*parts))
            tar.extract(member, dest or path.parent)


def extract_zip(path, drop_dirs: int = 0, dest: Path = None):
    with zipfile.ZipFile(path) as zf:
        for member in tqdm(zf.infolist(), desc=f'Uncompress {path}', unit='files', leave=False):
            p = Path(member.filename)
            parts = p.parts[drop_dirs:]
            if not parts:
                continue
            target = Path(*p.parts[drop_dirs:])
            if not member.is_dir():
                with zf.open(member.filename) as f:
                    tpath = Path(dest or path.parent, target)
                    tpath.parent.mkdir(parents=True, exist_ok=True)
                    with open(tpath, 'bw') as target:
                        target.write(f.read())


def zipdir(directory_path, zip_path, compression=zipfile.ZIP_DEFLATED):
    """
    Zip an entire directory and save it as a .zip file.
    """
    directory = Path(directory_path)
    with zipfile.ZipFile(zip_path, 'w', compression) as zipf:
        for file in directory.rglob('*'):
            arcname = file.relative_to(directory)
            zipf.write(file, arcname)


class Timer:
    def __init__(self):
        self._start = time.perf_counter()
        self._stop = None

    def reset(self) -> None:
        '''Reset the timer.'''
        self._start = time.perf_counter()

    def stop(self) -> None:
        '''Stop the timer. If the timer is already stopped then do nothing.'''
        if self._stop is None:
            self._stop = time.perf_counter()

    @property
    def time(self) -> float:
        '''Elapsed seconds since the Timer was created until now, or the time
        at which the timer was stopped.'''
        return (self._stop or time.perf_counter()) - self._start


@contextmanager
def catchtime(
    task_name='It', logger: Optional[Callable[[str], None]] = None, resolution=''
) -> Timer:
    '''
    Measure time elapsed from context entry to the time that stop is called.

    The timer is automatically stopped at context exit.

    During the context the time elapsed so far can also be queuried with `t.time`

    If logger is given then after context exit, the logger is called with the message

        "The {taskname} took {t.time:.2f} seconds

    If resolution is 'm' then milliseconds are used, 'u' micro, and 's' seconds.
    If empty or not given then the resolution is based on the length of the duration. Note this
    only affects the display, it does not impact the accuracy of the timer.

    Example::
        with catchtime('The network', LOG.debug, 'm') as t:
            # do something lengthy
        # equivalent to LOG.debug(f"The network took {t.time*1000:.2f} mseconds")
    '''
    valid = ('', 's', 'm', 'u')
    if logger and resolution not in valid:
        raise ValueError(f"Invalid resolution {resolution} for timer, valid are {valid}")
    t = Timer()
    try:
        yield t
    finally:
        t.stop()
        if logger:
            duration = t.time
            if not resolution:
                resolution = 's' if duration > 1 else 'm' if duration > 0.001 else 'u'
            if resolution == 's':
                logger(f'{task_name} took {t.time:.3f} seconds')
            elif resolution == 'm':
                logger(f'{task_name} took {t.time*1000:.3f} mseconds')
            elif resolution == 'u':
                logger(f'{task_name} took {t.time*1000000:.3f} useconds')


class SystemResourceMonitor:
    '''
    Monitor memory and CPU usage for a specified process.
    This is mostly for product validation and debugging.
    The CPU usage is the average CPU usage across all cores.
    '''

    def __init__(self, pid, interval=1):
        self.interval = interval
        self.process = psutil.Process(pid)
        self.running = False
        self.thread = None
        self.memory_snapshots = []
        self.cpu_snapshots = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def monitor(self):
        while self.running:
            current_memory = self.process.memory_info().rss / (1024**2)  # MB
            current_cpu = self.process.cpu_percent(interval=None)
            self.memory_snapshots.append(current_memory)
            self.cpu_snapshots.append(current_cpu)
            time.sleep(self.interval)

    @property
    def average_memory(self):
        return (
            sum(self.memory_snapshots) / len(self.memory_snapshots) if self.memory_snapshots else 0
        )

    @property
    def average_cpu(self):
        return sum(self.cpu_snapshots) / len(self.cpu_snapshots) if self.cpu_snapshots else 0

    @property
    def peak_memory(self):
        """Return the peak memory usage recorded."""
        return max(self.memory_snapshots, default=0)

    @property
    def peak_cpu(self):
        """Return the peak CPU usage recorded."""
        return max(self.cpu_snapshots, default=0)


@contextmanager
def catch_resources(
    pid=None, task_name='', logger: Optional[Callable[[str], None]] = None, interval=0.5
):
    '''
    Context manager to monitor memory usage during a block of code.

    Args:
        pid (int): The pid of the process to monitor. If None, the current process is monitored.
        task_name (str): The name of the task being monitored.
        logger (Callable[[str], None]): Optional logger function to output the results.
        interval (float): The time interval (in seconds) between memory checks.

    Example::
        with monitor_resources() as m:
            # do something lengthy
        print(f"Peak Memory Usage: {m.peak_memory:.2f} MB")
        print(f"Average Memory Usage: {m.average_memory:.2f} MB")
        print(f"Peak CPU Usage: {m.peak_cpu:.2f}%")
        print(f"Average CPU Usage: {m.average_cpu:.2f}%")
    '''
    pid = pid if pid is not None else os.getpid()
    monitor = SystemResourceMonitor(pid=pid, interval=interval)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()
        if logger:
            logger(f'[{task_name}] Peak Memory Usage: {monitor.peak_memory:.2f} MB')
            logger(f'[{task_name}] Average Memory Usage: {monitor.average_memory:.2f} MB')
            logger(f'[{task_name}] Peak CPU Usage: {monitor.peak_cpu:.2f}%')
            logger(f'[{task_name}] Average CPU Usage: {monitor.average_cpu:.2f}%')


class _FallbackAliveBar:
    def __enter__(self):
        return self

    def title(self, s):
        LOG.info(s)

    def __call__(self):
        pass

    def __exit__(self, *exc):
        pass


def spinner(total: Optional[int] = None):
    '''Return a spinner object that can be called to increment the count.

    It is based on alive_progress but with a fallback if alive_progress is not
    available.

    The returned object supports () to update position and `.title` to set the
    title text.
    '''
    try:
        import alive_progress
    except ImportError:
        # fallback until the cfg gets alive-progress
        return _FallbackAliveBar()
    else:
        # also hide the '0 in ' part if in spinner mode
        opts = dict(monitor='', elapsed='{elapsed}') if total is None else {}
        return alive_progress.alive_bar(total, enrich_print=False, stats=False, **opts)


class ExceptionThread(threading.Thread):
    '''Thread class with a join method that re-raises exceptions.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc = None

    def run(self):
        try:
            super().run()
        except logging_utils.UserError as e:
            self.exc = e
        except Exception as e:
            LOG.trace_exception()
            self.exc = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc


def load_labels(labels_path: str):
    '''Load labels from file, if provided.

    Supports three formats:
    1. Plain text with one label per line
    2. YAML files with a 'names' dictionary (e.g., names: {0: person, 1: bicycle})
    3. YAML files with a 'names' list (e.g., names: ['guns', 'knife'])
    '''
    if not labels_path:
        return []

    labels_path = Path(labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file {labels_path} not found")

    if labels_path.suffix.lower() in ('.yaml', '.yml'):
        with open(labels_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or 'names' not in data:
            raise ValueError(f"YAML file {labels_path} must contain a 'names' key")

        names = data['names']

        # Handle list format: names: ['guns', 'knife']
        if isinstance(names, list):
            return names

        # Handle dictionary format: names: {0: person, 1: bicycle}
        elif isinstance(names, dict):
            # Sort by numeric keys to maintain order
            return [
                names[i] for i in sorted(int(k) if isinstance(k, str) else k for k in names.keys())
            ]

        else:
            raise ValueError(f"'names' in {labels_path} must be either a list or dictionary")

    # Handle plain text files (one label per line)
    return [x for x in labels_path.read_text().splitlines() if x]


class LimitedLengthDataLoader:
    '''Wrapper to limit the length of a dataloader.'''

    def __init__(self, dataloader, max_length):
        self.sampler = None
        self.dataloader = dataloader
        self.max_length = max_length

    def __len__(self):
        return min(self.max_length, len(self.dataloader))

    def __iter__(self):
        import itertools

        return itertools.islice(self.dataloader, self.max_length)


def path_sanity_check(path) -> None:
    if not os.path.exists(path):
        raise ValueError("The root directory does not exist.")

    if not os.listdir(path):
        raise ValueError("The root directory is empty.")

    if not os.path.isdir(path):
        raise ValueError("The root directory is not a directory.")


def _exec_grep(cmd, detect, matchers, no_match, log_level=logging.DEBUG):
    try:
        LOG.debug("$ %s", cmd)
        p = subprocess.run([cmd], encoding='utf8', check=True, capture_output=True, shell=True)
        lines = p.stdout.splitlines()
        matching, values = [], []
        for match, value in matchers:
            if matched := [l for l in lines if re.search(match, l)]:
                matching.extend(matched)
                values.append(value)

        matching_str = '\n'.join(matching)
        if matching:
            LOG.trace(f"Found {detect} in {cmd} output: {matching_str}")
        else:
            lines = '\n'.join(lines)
            LOG.log(log_level, f"Did not find {detect} in {cmd} output")
            LOG.trace(f"{cmd} output : {lines}")
            return no_match
        if len(values) > 1 and any(values[0] != v for v in values[1:]):
            LOG.warning(f"Multiple matches for {detect} in {cmd} output: {matching_str}")
            return no_match
        return values[0]
    except subprocess.CalledProcessError as e:
        LOG.log(log_level, f"Could not exec {cmd}: {e}")
        LOG.debug(e.stderr)
        return no_match


def is_vaapi_available():
    if platform.uname().machine != 'x86_64':
        return False
    level = logging.INFO
    return _exec_grep(
        'vainfo', 'VA-API', [('vainfo: Supported profile and entrypoints', True)], False, level
    )


def is_opencl_available():
    CHECK_DOCS = "Please check the documentation for installation instructions"
    try:
        import pyopencl

    except ImportError as e:
        LOG.warning(f"pyopencl not installed/imported, could not detect OpenCL platforms : {e}")
        return False
    try:
        platforms = pyopencl.get_platforms()
    except Exception as e:
        LOG.warning(f"Failed to get OpenCL platforms : {e}")
        LOG.warning(CHECK_DOCS)
        return False
    else:
        if not platforms:
            LOG.warning("No OpenCL platforms found")
            LOG.warning(CHECK_DOCS)
            return False
        for p in platforms:
            try:
                if devs := p.get_devices(pyopencl.device_type.GPU):
                    sdevs = ', '.join(d.name for d in devs)
                    LOG.debug(f"Found OpenCL GPU devices for platform {p.name}: {sdevs}")
                    return True

                LOG.debug(f"No GPU OpenCL devices in platform {p.name}")
            except Exception as e:
                LOG.warning(f"Failed to get OpenCL devices for platform {p.name} : {e}")
        LOG.warning("No OpenCL GPU devices found")
        LOG.warning(CHECK_DOCS)
        return False


def get_backend_opengl_version(backend, default=env.DEFAULTS.opengl_backend):
    backend = backend.lower()
    if backend != default:
        LOG.info(f"Default OpenGL backend {default} overridden, using {backend}")
    try:
        api, major, minor = backend.split(",")
        if api not in ['gles', 'gl']:
            raise ValueError()
        return (api, int(major), int(minor))
    except (ValueError, TypeError):
        LOG.error(f"Invalid OpenGL backend {backend}, using default ({default})")
        api, major, minor = default.split(",")
        return (api.lower(), int(major), int(minor))


def is_opengl_available(backend):
    api, _, _ = get_backend_opengl_version(backend)
    try:
        import pyglet

    except ImportError as e:
        LOG.warning(f"pyglet not installed, could not detect OpenGL availability: {e}")
        return False

    # If using GL ES, we can only check import, as interacting with pyglet subpackages
    # is unsafe in GL ES without additional configuration, which is not performed here.
    if api == "gles":
        return True
    try:
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        screen.get_best_config()
    except Exception as e:
        LOG.warning(f"pyglet could not access the display, OpenGL is not available: {e}")
        return False
    return True


def is_compiler_available():
    with open(os.devnull, 'w') as devnull:
        try:
            _saved_out, _saved_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            from axelera import compiler

            return True
        except ImportError:
            return False
        finally:
            sys.stdout, sys.stderr = _saved_out, _saved_err


def ensure_dependencies_are_installed(dependencies: list[str], dry_run: bool = False):
    '''Ensure that the dependencies are installed.

    First checks which dependencies are missing, then installs them with a progress bar.
    '''
    if not dependencies:
        return

    def _sanitize(dep):
        # expand any env vars (for -r)
        # convert any space-separated -r to single-token form for passing in list
        return re.sub(r'-r\s+', '-r', os.path.expandvars(dep))

    dependencies = [_sanitize(dep) for dep in dependencies]

    # First check what's missing using dry-run
    try:
        result = subprocess.run(
            ["pip", "install", "--dry-run"] + dependencies,
            encoding='utf8',
            check=True,
            capture_output=True,
        )

        # Check if there are any packages to collect
        packages_to_install = []
        for line in result.stdout.splitlines():
            if line.startswith("Collecting "):
                packages_to_install.append(line.split()[1])

        if not packages_to_install:
            LOG.debug("All dependencies are already installed")
            return

        LOG.info(f"Missing dependencies that will be installed: {', '.join(packages_to_install)}")

        if dry_run:
            return

        # Install missing dependencies with progress
        cmd = ["pip", "install"] + dependencies
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )

        # Print output in real-time with flush
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            print(line, end='', flush=True)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        msg = '\n'.join(
            [' '.join(["pip", "install", "--dry-run"] + dependencies), e.stdout, e.stderr]
        )
        raise RuntimeError(f"Failed to check dependencies requested in the network yaml:\n{msg}")


class FrozenIntEnumMeta(EnumMeta):
    "Enum metaclass that freezes an enum entirely"

    def __new__(mcls, name, bases, classdict):
        classdict['__frozenenummeta_creating_class__'] = True
        enum = super().__new__(mcls, name, bases, classdict)
        del enum.__frozenenummeta_creating_class__
        return enum

    def __call__(cls, value, names=None, *, module=None, **kwargs):
        if names is None:  # simple value lookup
            return cls.__new__(cls, value)
        enum = IntEnum._create_(value, names, module=module, **kwargs)
        enum.__class__ = type(cls)
        return enum

    def __setattr__(cls, name, value):
        members = cls.__dict__.get('_member_map_', {})
        if hasattr(cls, '__frozenenummeta_creating_class__') or name in members:
            return super().__setattr__(name, value)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))

    def __delattr__(cls, name):
        members = cls.__dict__.get('_member_map_', {})
        if hasattr(cls, '__frozenenummeta_creating_class__') or name in members:
            return super().__delattr__(name)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))


class FrozenIntEnum(IntEnum, metaclass=FrozenIntEnumMeta):
    pass


def ident(x):
    return re.sub(r'\W+', '_', x)


def create_enumerators(labels):
    '''
    Create enumerators from a list of labels. ', ' separated labels may be given,
    and will be treated as aliases for the first label in the list. If there are
    multiple classes with the same label, we attempt to use an alias if available.
    Raises a ValueError if there are no unique aliases available for this substituion.
    Duplicate aliases will be ignored, and if a class requires an existing alias as its
    base name, this will take priority over the alias.
    '''
    bases = {}
    aliases = {}
    for i in range(len(labels)):
        base, *other = [ident(l).lower() for l in labels[i].split(", ")]

        if (existing := bases.get(base, None)) is not None:
            if sub := [s for s in other if s not in bases]:
                LOG.trace(
                    f"Dataset enumeration: Replacing class {i} label '{base}' with alias '{sub[0]}', as '{base}' already a base name for class {existing}"
                )
                base = sub[0]
                other.remove(sub[0])
            else:
                # before failing, see if the existing class has an alias
                # not already used as a base, so we can free up the base
                # name for this class. Note we could do this recursively
                # if the alias is used as a base by a further different class,
                # for smarter resolution, but this is probably overkill so
                # not implemented here.
                if sub := [
                    s for s, cls_id in aliases.items() if cls_id == existing and s not in bases
                ]:
                    bases[sub[0]] = existing
                    del aliases[sub[0]]
                    LOG.trace(
                        f"Dataset enumeration: Replacing class {existing} label '{base}' with alias '{sub[0]}', as '{base}' already a base name for class {i}"
                    )
                else:
                    raise ValueError(
                        f"Class {i} label '{base}' is a duplicate base name for class {existing}, and no suitable alias found"
                    )
        # base names take priority over aliases
        if (existing := aliases.get(base, None)) is not None:
            LOG.trace(
                f"Dataset enumeration: Skipping class {existing} alias '{base}', as it is a base name for class {i}"
            )
            del aliases[base]

        bases[base] = i

        for alias in other:
            if (existing := bases.get(alias, None)) is not None or (
                existing := aliases.get(alias, None)
            ) is not None:
                LOG.trace(
                    f"Dataset enumeration: Skipping class {i} alias {alias} as it is already used for class {existing}"
                )
            else:
                aliases[alias] = i
    return list(bases.items()) + list(aliases.items())
