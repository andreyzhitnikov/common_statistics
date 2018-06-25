import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="common_statistic ",
    version="0.0.1",
    author="Andrey, Rotem",
    author_email="andrey@campus.technion.ac.il",
    description="Package for paper: Revealing Common Statistical Behaviors in Heterogeneous Populations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://andreeyz@bitbucket.org/andreeyz/revealing-common-statistical-behaviors-in-heterogeneous.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
