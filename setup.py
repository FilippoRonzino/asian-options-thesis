from setuptools import setup, find_packages

setup(
    name="bsc-thesis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "seaborn",
        "tensorflow",
        "torch",
    ],
)