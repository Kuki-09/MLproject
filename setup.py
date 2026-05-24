from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return dependencies as a list
    """

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="student-performance-prediction",
    version="0.0.1",
    author="Hiteshi Kukreja",
    author_email="hiteshi724@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)