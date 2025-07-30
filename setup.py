from setuptools import find_packages, setup

setup(
    name="mindbug",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
)
