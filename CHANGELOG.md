# Changelog

All notable changes to this SDK will be documented in this file to assist users in migrating their YAML configurations and pipelines in line with SDK upgrades, ensuring seamless model and pipeline deployment.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Breaking Changes
- To deploy an ONNX model, update the YAML `models` section to use the new path:  
  `$AXELERA_FRAMEWORK/ax_models/base_onnx.py`  
  instead of:  
  `$AXELERA_FRAMEWORK/ax_models/onnx/model.py`.  
  - This change aligns the ONNX model deployment path with the structure used for `base_torch.py`.
- Added variable substitution support for Axelera YAMLs, with the syntax
  being ${{MODEL_INFO_VARIABLE}}
  - Any templates using the old {{MODEL_INFO_VARIABLE}} syntax must prepend
    the `$` to continue working

### Added
- Introduced strictYAML and built a schema for built-in AxOperators to provide clear messages
  for incorrect usage in the YAML pipeline
- display.Window now has a title method that allows a stream to be identified.
  inference.py will set the title if there is more than one stream.

### Changed
- 

### Deprecated
- Features that will be removed in upcoming releases

### Removed
- Features removed in this release

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes

## [1.0.0] - 2024-11-01
### Added
- Initial release

## Usage
- Start with the [README](/README.md)

### Pull Request Process
1. Every PR should consider updating `CHANGELOG.md`.
2. Add changes under the `[Unreleased]` section.
3. Use appropriate categories.
4. For breaking changes:
   - Add under the "Breaking Changes" category.
   - Include migration instructions if possible.

### Release Process
1. When ready for release:
   - Create a new version section (e.g., `[2.0.0]`).
   - Move items from `[Unreleased]` to the new version.
   - Add the release date.
2. Branch out from `main` as `release/<version>` (e.g., `release/2.0.0`)
3. Remove the `[Unreleased]`, `[Pull Request Process]`, and `[Release Process]` sections.
