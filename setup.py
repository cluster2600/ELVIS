from setuptools import setup, find_packages

setup(
    name="trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "rich",
        "pyyaml",
        "ccxt",
        "ta",
    ],
    python_requires=">=3.10",
) 