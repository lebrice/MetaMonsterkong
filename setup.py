from setuptools import setup, find_packages
packages = find_packages()
import os
from typing import List

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")


def load_requirements(requirements_path = "requirements.txt") -> List[str]:
    with open(requirements_path, "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
        lines = [line for line in lines if not line.startswith("#")]
        lines = [line for line in lines if "git+" not in line]
    return lines


def load_github_requirements(requirements_path = "requirements.txt") -> List[str]:
    with open(requirements_path, "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
        lines = [line for line in lines if not line.startswith("#")]
        lines = [line for line in lines if "https://" in line]
    return lines


setup(name='meta_monsterkong',
    version='0.0.1',
    description="meta_monsterkong: samples a new map uniformly at random from a directory of generated maps",
    install_requires=load_requirements(requirements_path),
    dependency_links=load_github_requirements(requirements_path),
    packages = packages,
    include_package_data = True,
    package_data = {
        # Include all the assets and maps.
        # IDEA: Could be cool to generate the maps after install, rather than bundling
        # them together with the package.
        "meta_monsterkong": ["assets/*", "maps/*"],
    },
    python_requires=">=3.6",
)
