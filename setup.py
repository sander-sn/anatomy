from setuptools import setup

setup(
    name="anatomy",
    version="0.1.0",
    packages=["anatomy"],
    install_requires=[
        "joblib>=0.15.0",
        "numpy>=1.18.1",
        "pandas>=1.0.0"
    ],
    python_requires=">=3.9",
    url="https://github.com/sander-sn/anatomy",
    author="Sander Schwenk-Nebbe",
    author_email="sandersn@econ.au.dk",
    license="MIT"
)
