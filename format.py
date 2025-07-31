#!/usr/bin/env python3
"""Format all Python files with Black."""

import subprocess
import sys

# Run Black on all Python files
result = subprocess.run([sys.executable, "-m", "black", "src", "tests"], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

sys.exit(result.returncode)