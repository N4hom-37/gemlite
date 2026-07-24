from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Tells setuptools this package contains a platform-specific binary
    (the prebuilt Rust cdylib under lib/), even though it isn't compiled
    by setuptools itself. Without this, bdist_wheel emits a pure "py3-none-any"
    wheel and ignores --plat-name, which breaks the per-platform tagging
    used in .github/workflows/main.yml.
    """

    def has_ext_modules(self) -> bool:
        return True


class build_py(_build_py):
    """package-dir maps gemlite straight to the repo root (to keep setup.py,
    Cargo.toml, and the Python package side by side), so the default module
    scan would also sweep up setup.py itself as "gemlite.setup". Filter it out.
    """

    def find_package_modules(self, package, package_dir):
        return [
            m for m in super().find_package_modules(package, package_dir)
            if m[1] != "setup"
        ]


setup(distclass=BinaryDistribution, cmdclass={"build_py": build_py})
