#!/usr/bin/env python3
"""
Scans ApraPipes headers for Module subclasses.
Used to detect modules that haven't been registered with the declarative pipeline system.

Usage:
    python scan_modules.py <include_dir> [--json] [--exclude-abstract]

Output:
    List of module class names found in headers (one per line, or JSON array)
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Set, List

# Modules that are abstract base classes and shouldn't be registered
ABSTRACT_MODULES = {
    "Module",           # Base class itself
    "AbsControlModule", # Abstract control module base
}

# Pattern to match class declarations inheriting from Module or *Module
MODULE_PATTERN = re.compile(
    r'class\s+(\w+)\s*:\s*(?:virtual\s+)?public\s+(?:virtual\s+)?(\w*Module\w*)',
    re.MULTILINE
)

def scan_header_file(filepath: Path) -> Set[str]:
    """Scan a single header file for Module subclasses."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        matches = MODULE_PATTERN.findall(content)
        return {class_name for class_name, base_class in matches
                if base_class.endswith('Module') or base_class == 'Module'}
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return set()

def scan_directory(include_dir: Path, exclude_abstract: bool = True) -> List[str]:
    """Scan all header files in directory for Module subclasses."""
    all_modules: Set[str] = set()

    # Scan all .h files
    for header_file in include_dir.rglob("*.h"):
        # Skip declarative infrastructure files
        if "declarative" in str(header_file):
            continue
        modules = scan_header_file(header_file)
        all_modules.update(modules)

    # Remove abstract modules if requested
    if exclude_abstract:
        all_modules -= ABSTRACT_MODULES

    return sorted(all_modules)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <include_dir> [--json] [--exclude-abstract]",
              file=sys.stderr)
        sys.exit(1)

    include_dir = Path(sys.argv[1])
    if not include_dir.is_dir():
        print(f"Error: {include_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    use_json = "--json" in sys.argv
    exclude_abstract = "--exclude-abstract" in sys.argv or "--include-abstract" not in sys.argv

    modules = scan_directory(include_dir, exclude_abstract)

    if use_json:
        print(json.dumps(modules))
    else:
        for module in modules:
            print(module)

if __name__ == "__main__":
    main()
