<!---
SPDX-FileCopyrightText: Copyright-2025, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
SPDX-License-Identifier: LicenseRef-LICENSE
--->
# Building Wheel Variants

This document describes how to build multiple wheel variants with different custom configurations.

## Overview

The MLIA package can be built with different custom configurations that customize the performance estimator settings and target profiles. The build system supports:

* Building a default wheel without any custom configuration
* Building multiple variant wheels with custom configurations
* Automatic restoration of the repository state after builds
* Configurable source of custom configurations
* Custom configurations are **added** to defaults, not replaced

## Quick Start

### List Available Variants

To see which configuration variants are available:

```bash
# Using tox
tox -e build-variants -- --list-variants

# Or directly with Python
python build_tools/variant_builder.py --list-variants
```

### Build All Variants

To build the default wheel plus all available custom variants:

```bash
# From default configuration dir
tox -e build-variants

# Or from custom configuration dir
tox -e build-variants -- --config-dir path_to_dir
```

This will:

* Build the default wheel (without any variant tag)
* Build a wheel for each custom configuration found
* Clean build cache between variants to ensure isolation
* Restore the repository to its original state
* Place all wheels in the `dist/` directory

Each variant wheel will be tagged with the variant name, e.g., `mlia-1.0.0.custom-variant-py3-none-manylinux2014_x86_64.whl`

### Build Specific Variants

To build only specific variants:

```bash
# Build only specific variants (plus default)
tox -e build-variants -- --variants variant1 variant2

# Build only variants without the default wheel
tox -e build-variants -- --variants variant1 variant2 --no-default
```

### Build for Different Platforms

To build for a specific platform (default is `manylinux2014_x86_64`):

```bash
# Build for aarch64
tox -e build-variants -- --platform manylinux2014_aarch64

# Or use the direct script
python build_tools/variant_builder.py --platform manylinux2014_aarch64
```

## Configuration Structure

Custom configurations are expected to be located in `../mlia-tools/mlia-gc-sys-configs/` by default, otherwise the location has to be specified with `--config-dir`. Each variant **must** have a `variant_config.json` file that defines where files should be copied.

### Directory Structure

```
mlia-gc-sys-configs/
├── custom-variant-1/
│   ├── variant_config.json      # Configuration file defining copy operations
│   ├── configurations/          # Source files for nx-performance-estimator
│   │   ├── system-config-*.ini
│   │   └── compiler-*.ini
│   └── target_profile/          # Source files for target profiles
│       └── *.toml
├── custom-variant-2/
│   ├── variant_config.json
│   ├── configurations/
│   └── target_profile/
└── ...
```

### variant_config.json Format

Each variant directory must contain a `variant_config.json` file with the following structure:

```json
{
  "name": "custom-variant",
  "description": "Custom variant configuration",
  "copy_paths": [
    {
      "src": "configurations",
      "dst": "src/mlia/resources/nx-performance-estimator/configurations"
    },
    {
      "src": "target_profile",
      "dst": "src/mlia/resources/target_profiles"
    }
  ]
}
```

**Fields:**

* `name` (optional): Human-readable name for the variant
* `description` (optional): Description of the variant purpose
* `copy_paths` (required): Array of copy operations
   * `src`: Source path relative to the variant directory
   * `dst`: Destination path relative to the mlia repository root

**Important:** Files specified in `copy_paths` are **added** to the existing default files, not replaced.
Each variant wheel contains:

* All default MLIA configurations
* Plus the custom configurations for that variant
