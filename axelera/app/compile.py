# Copyright Axelera AI, 2024
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from axelera import types

from . import config, constants, exceptions, logging_utils, utils

if TYPE_CHECKING:
    from axelera.compiler.quantized_model import AxeleraQuantizedModel

LOG = logging_utils.getLogger(__name__)
ASSET_DIR_STR = "assets"  # for saving downloaded data


def get_padded_low_high(n_padded_ch_inputs, tensor_layout, where='H'):
    if where not in tensor_layout:
        raise ValueError(
            f"'where' must be a character present in 'tensor_layout'. Got {where} not in {tensor_layout}"
        )

    low_index = tensor_layout.index(where) * 2
    high_index = low_index + 1

    result = tuple((inputs[low_index], inputs[high_index]) for inputs in n_padded_ch_inputs)

    return result


def get_original_shape(
    output_shapes: Tuple[Tuple[int, ...], ...],
    n_padded_ch: Tuple[Tuple[int, ...], ...],
    current_layout: str,
    expected_layout: Optional[str] = None,
) -> List[Tuple[int, ...]]:
    if expected_layout is None:
        expected_layout = current_layout

    if len(output_shapes) != len(n_padded_ch):
        raise ValueError("Output shapes and number of padded channels must have the same length")

    if expected_layout:
        for dim in expected_layout:
            if dim not in current_layout:
                raise ValueError(
                    f"Dimension {dim} in expected layout is not present in current layout"
                )

    adjusted_shapes = []
    for shape, padding in zip(output_shapes, n_padded_ch):
        if len(shape) != len(current_layout):
            raise ValueError("All dimensions in output shapes must be present in current layout")
        if len(padding) != 2 * len(current_layout):
            raise ValueError("Padding information must be twice the length of current layout")

        adjusted_shape = list(shape)
        for dim, pad_low, pad_high in zip(current_layout, padding[::2], padding[1::2]):
            dim_idx = current_layout.index(dim)
            adjusted_shape[dim_idx] -= pad_low + pad_high

        adjusted_shapes.append(tuple(adjusted_shape))

    if expected_layout and current_layout != expected_layout:  # Need to convert layout
        dim_mapping = {dim: idx for idx, dim in enumerate(current_layout)}
        permuted_order = [dim_mapping[dim] for dim in expected_layout if dim in dim_mapping]
        adjusted_shapes = [
            tuple(shape[idx] for idx in permuted_order) for shape in adjusted_shapes
        ]

    return adjusted_shapes


def load_manifest_from_file(manifest_json: Path) -> types.Manifest:
    with open(file=manifest_json, mode="r") as f:
        data: dict = json.load(f)
    return types.Manifest(**data)


def load_prequant_manifest(output_path: Path) -> types.Manifest:
    # Prequantized TVM model must have the following 2 files
    quantized_model_json = output_path / constants.K_MODEL_QUANTIZED_FILE_NAME
    manifest_json = output_path / constants.K_MANIFEST_FILE_NAME
    if not quantized_model_json.exists():
        raise FileNotFoundError(f"Failed to find {quantized_model_json}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"Failed to find {manifest_json}")
    with open(manifest_json) as quantized_meta:
        params = json.load(quantized_meta)

    # check if manifest version is compatible; if not, warn the user for now
    # TODO: we probably should requantize the model?
    manifest_version = params.get('manifest_version', '')
    expected_version = types.Manifest.__dataclass_fields__['manifest_version'].default
    if manifest_version != expected_version:
        LOG.warning(
            f"The prequantized manifest version: {manifest_version} may not be "
            f"compatible with the new version: {expected_version}"
        )

    # revise the path according to the output_path
    for file in [
        'model_lib_file',
        'preprocess_graph',
        'postprocess_graph',
        'quantized_model_file',
    ]:
        if params[file]:
            params[file] = str(output_path / Path(params[file]).name)
    return types.Manifest(**params)


def _download_prequant(
    model_info: types.ModelInfo, asset_path: Path, delete_archive: bool = True
) -> Tuple[types.Manifest, bool]:
    # assume the file name is the last part of the path
    file_path = urlparse(model_info.prequantized_url).path
    target_path = asset_path / Path(file_path).name
    utils.download_and_extract_asset(
        model_info.prequantized_url,
        target_path,
        model_info.prequantized_md5,
        delete_archive=delete_archive,
    )
    return target_path


def download_prequant_and_build_manifest(
    model_info: types.ModelInfo, output_path: Path
) -> Tuple[types.Manifest, Path]:
    '''Download pre-quantized model from Axelera Cloud and build the manifest instance'''
    asset_path = output_path / ASSET_DIR_STR
    quantized_dir = output_path / constants.K_MODEL_QUANTIZED_DIR
    _download_prequant(model_info, asset_path)
    manifest_file = asset_path / constants.K_MANIFEST_FILE_NAME
    bad_path = asset_path / "build" / model_info.name / "x86_64" / model_info.name
    if not manifest_file.exists():
        raise RuntimeError(f"Unrecognised prequantized files from {asset_path}")

    if not quantized_dir.exists():
        LOG.debug(f'moving files from {asset_path} to {quantized_dir}')
        shutil.copytree(asset_path, quantized_dir, copy_function=shutil.move, dirs_exist_ok=True)
    else:
        LOG.debug(f"quantized dir already exists - not overwriting: {quantized_dir}")
    shutil.rmtree(asset_path)
    return load_prequant_manifest(quantized_dir), quantized_dir


class ManifestEncoder(json.JSONEncoder):
    """Custom JSON encoder for Manifest class"""

    def default(self, obj):
        import tvm

        if isinstance(obj, tvm.tir.IntImm):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tvm.ir.container.Array) or isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        return super().default(obj)


def load_quantized_model_from_manifest(manifest: types.Manifest) -> "AxeleraQuantizedModel":
    from axelera.compiler import quantized_model

    return quantized_model.AxeleraQuantizedModel.load(manifest)


def _backup_and_load_quantized(tmp_dir: Path, manifest: types.Manifest):
    """back up the pre/postprocess graphs so the compiler will not remove them,
    then load the quantized model"""
    graph_paths = []
    for graph in [manifest.preprocess_graph, manifest.postprocess_graph]:
        if graph:
            graph_path = Path(graph)
            new_file_path = tmp_dir / graph_path.name
            LOG.debug(f"Backing up {graph_path} to {new_file_path}")
            shutil.copy(graph_path, new_file_path)
            graph_paths.append(new_file_path)
        else:
            graph_paths.append(graph)
    manifest.preprocess_graph, manifest.postprocess_graph = graph_paths

    try:
        quant_model = load_quantized_model_from_manifest(manifest)
    except ValueError as e:
        if "Cannot update from version" in str(e):
            version = str(e).split()[-1]
            raise ValueError(
                f"The pre-quantized model version ({version}) is not compatible with the current compiler version. "
                "Please run deploy.py with --mode=quantcompile"
            ) from e
        raise

    return manifest, quant_model


def prepare_build_directory(output_path: Path, deploy_mode: config.DeployMode):
    """Remove all files and subdirectories in the given directory if it exists;
    otherwise, create the directory."""

    LOG.debug(f"Preparing build directory: {output_path}")

    def _clear_dir(dir_path):
        for item in dir_path.iterdir():
            if item.is_dir():
                _clear_dir(item)
                item.rmdir()
            else:
                item.unlink()

    if deploy_mode == config.DeployMode.QUANTIZE:
        manage_quantized_directory(output_path)

    if output_path.is_dir():
        _clear_dir(output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)


def manage_quantized_directory(dir_path: Path):
    if not dir_path.is_dir():
        return

    existing_dirs = sorted(
        [
            p
            for p in dir_path.parent.glob(f"{dir_path.name}_*")
            if p.is_dir() and p.name[len(dir_path.name) + 1 :].isdigit()
        ],
        key=lambda p: int(p.name[len(dir_path.name) + 1 :]),
        reverse=True,
    )

    # Move existing directories down the sequence
    for existing_dir in existing_dirs:
        counter = int(existing_dir.name[len(dir_path.name) + 1 :])
        if counter > 9:
            LOG.warning(f"There are over 10 quantized dir, removing {existing_dir}")
            shutil.rmtree(str(existing_dir))
        else:
            new_dir = Path(f"{dir_path}_{counter+1}")
            shutil.move(str(existing_dir), str(new_dir))

    # Move the main directory
    new_dir_path = Path(f"{dir_path}_1")
    shutil.move(str(dir_path), str(new_dir_path))


def _write_manifest(manifest, output_path):
    manifest_json = output_path / constants.K_MANIFEST_FILE_NAME
    j = json.dumps(manifest.__dict__, indent=2, cls=ManifestEncoder)
    manifest_json.write_text(j)


def _log_config(compilation_cfg):
    LOG.debug(f"Compilation config: {json.dumps(compilation_cfg.as_dict(), indent=2)}")


# This is a slightly brittle way to get some progress out of the compiler. It relies on the
# compiler logging the phase it is in. If the messages change, it just means the progress bar will
# be off by one phase, but as we also have a spinner and a timer, you at least still get some
# feedback.
COMPILER_PHASES = (
    "Running LowerFrontend",
    "Running FrontendToMidend",
    "Running LowerMidend",
    "Running MidendToTIR",
    "Running TIRToRuntime",
    "axe_module_save_binary called",
    "Lowering finished!",
)
COMPILER_LOGGERS = ["te_compiler", "onnx2torch", "compiler", "axelera.compiler"]


@contextlib.contextmanager
def capture_progress():
    loggers = [logging.getLogger(x) for x in COMPILER_LOGGERS]

    # if -v is given then show 0/7 phases, otherwise just show the spinner
    desired_level = loggers[0].getEffectiveLevel()
    phases = len(COMPILER_PHASES) if desired_level <= logging.INFO else None
    with utils.spinner(phases) as bar:

        class _Handler(logging.StreamHandler):
            def __init__(self):
                super().__init__()

            def filter(self, record: logging.LogRecord) -> str:
                msg = record.getMessage()
                if msg.startswith(COMPILER_PHASES):
                    phase = [n for n, p in enumerate(COMPILER_PHASES) if msg.startswith(p)][0]
                    bar.title(COMPILER_PHASES[phase])
                    bar()
                    return False
                return True

        handler = _Handler()
        for logger in loggers:
            logger.addHandler(handler)
        try:
            yield
        finally:
            for logger in loggers:
                logger.removeHandler(handler)


def _make_manifest_relative(manifest: types.Manifest, relative_to: Path):
    for field in [
        'model_lib_file',
        'model_params_file',
        'quantized_model_file',
        'preprocess_graph',
        'postprocess_graph',
    ]:
        if val := getattr(manifest, field):
            manifest.__dict__[field] = str(Path(val).relative_to(relative_to))


def _compile_quantized(
    tmp_dir,
    preloaded_manifest,
    quantized_dir,
    compilation_cfg,
    output_path,
):
    from axelera.compiler import top_level

    prepare_build_directory(output_path, None)
    preloaded_manifest, quant_model = _backup_and_load_quantized(tmp_dir, preloaded_manifest)

    # after top_level.compile, output_path will be cleared and filled by the compiler
    with capture_progress():
        try:
            the_manifest = top_level.compile(quant_model, compilation_cfg, output_path)
        except Exception as e:
            import traceback

            LOG.error(traceback.format_exc())
            raise e
        _make_manifest_relative(the_manifest, output_path)
    LOG.debug(f"Using prequantized model from: {quantized_dir}")
    return preloaded_manifest, the_manifest


def _check_and_use_local_prequant(tmp_dir, quantized_dir, compilation_cfg, output_path):
    LOG.debug(f"Checking for prequantized model in: {quantized_dir}")
    try:
        preloaded_manifest = load_prequant_manifest(quantized_dir)
    except FileNotFoundError as e:
        LOG.debug(f"Failed to load prequantized model: {e}")
        return False, None, None

    preloaded_manifest, the_manifest = _compile_quantized(
        tmp_dir,
        preloaded_manifest,
        quantized_dir,
        compilation_cfg,
        output_path,
    )
    return True, preloaded_manifest, the_manifest


def _delete_tvm_codegen_files(output_path: Path):
    for pattern in [
        'axelera_module.so',
        'copy_cmd_list.txt',
        'graph.json',
        'kernel.c',
        'params.bin',
        'tvmgen_*.*',
        'lib.so',
    ]:
        for p in output_path.glob(pattern):
            LOG.trace(f"Removing {p}")
            p.unlink()


def compile(
    model: types.Model,
    model_info: types.ModelInfo,
    compilation_cfg,  # axelera.compiler.Configuration
    model_dir: Path,
    is_export: bool,
    deploy_mode: config.DeployMode,
    metis: config.Metis,
    decoration_flags: str,
) -> types.Manifest:
    from axelera.compiler import top_level

    model_name = model_info.name

    if deploy_mode not in {config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG}:
        output_path = model_dir / decoration_flags
    else:
        output_path = model_dir / constants.K_MODEL_QUANTIZED_DIR

    LOG.debug(f"output path: {output_path}\ndeploy mode: {deploy_mode}")

    quantized_dir = model_dir.joinpath(constants.K_MODEL_QUANTIZED_DIR)
    tmp_dir = Path(tempfile.mkdtemp())
    preloaded_manifest = None

    if deploy_mode in {
        config.DeployMode.QUANTIZE,
        config.DeployMode.QUANTIZE_DEBUG,
        config.DeployMode.QUANTCOMPILE,
    }:
        prepare_build_directory(output_path, deploy_mode)
        if compilation_cfg.quantization_config.get("quantization_debug"):
            compilation_cfg.quantization_config["quantized_model_debug_save_dir"] = str(
                output_path
            )
        _log_config(compilation_cfg)
        # with capture_progress():  TODO make the spinner interact with tqdm nicely
        try:
            the_manifest = top_level.compile(model, compilation_cfg, output_path)
        except Exception as e:
            import traceback

            LOG.error(traceback.format_exc())
            raise e
        _make_manifest_relative(the_manifest, output_path)

        if deploy_mode == config.DeployMode.QUANTCOMPILE:
            # we don't need _preloaded_manifest but we want the pre/post graphs
            # to be backed up and kept track of
            _write_manifest(the_manifest, output_path)
            _preloaded_manifest = load_prequant_manifest(output_path)
            _backup_and_load_quantized(tmp_dir, _preloaded_manifest)

    elif deploy_mode == config.DeployMode.PREQUANTIZED:
        # TODO: probably should check if the config is different from the quantized one
        found_prequant, preloaded_manifest, the_manifest = _check_and_use_local_prequant(
            tmp_dir,
            quantized_dir,
            compilation_cfg,
            output_path,
        )

        if not found_prequant and model_info.prequantized_url:
            LOG.debug(
                f"Downloading specified prequantized model from: {model_info.prequantized_url}"
            )
            preloaded_manifest, quantized_dir = download_prequant_and_build_manifest(
                model_info, output_path.parent.parent
            )
            preloaded_manifest, the_manifest = _compile_quantized(
                tmp_dir,
                preloaded_manifest,
                quantized_dir,
                compilation_cfg,
                output_path,
            )
            found_prequant = True

        if not found_prequant:
            raise exceptions.PrequantizedModelRequired(model_name, quantized_dir)

        LOG.debug(f"Using prequantized model compiled to: {output_path}")
    _delete_tvm_codegen_files(output_path)
    _write_manifest(the_manifest, output_path)
    shutil.rmtree(tmp_dir)

    # save config to the build folder, we need to know
    # how the model is compiled when building the pipeline
    config_json = output_path / constants.K_COMPILE_CONFIG_FILE_NAME
    j = json.dumps(compilation_cfg.as_dict(), indent=2)
    config_json.write_text(j)

    if is_export:
        # package the build folder into a zip file
        exported_root = config.default_exported_root()
        exported_root.mkdir(parents=True, exist_ok=True)
        suffix = {config.DeployMode.QUANTIZE: '-prequantized'}.get(deploy_mode, '')
        export_path = exported_root / f"{model_name}{suffix}.zip"
        utils.zipdir(output_path.parent.parent, export_path)
        LOG.info(f"Exported to {export_path}")
    return the_manifest
