# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Neural Technology standardized-output filtering defaults."""

from mlia.core.settings import CollapseRule

DEFAULT_COLLAPSE_RULES = (
    CollapseRule(
        kind="code_stack",
        attribute="file",
        globs=(
            "**/site-packages/torch/**",
            "**/site-packages/executorch/**",
        ),
    ),
)
