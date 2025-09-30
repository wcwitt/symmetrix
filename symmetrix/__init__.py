from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("symmetrix")
except PackageNotFoundError as e:
    # package is not installed
    __version__ = "0.0.0"