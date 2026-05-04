"""
Compatibility shim for Python 3.12+ where `distutils` was removed.

Some third-party packages (e.g., certain RVC CLI wrappers) still import
`distutils.util.strtobool`. We provide the minimal surface area needed.
"""

