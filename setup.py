from setuptools import setup, find_packages

setup(
    name="msp_da",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
    ],
)
