from setuptools import setup, find_packages

setup(
    name="hf_ehr",
    version="0.1.1",
    author="Michael Wornow",
    packages=find_packages(),
    description="Code for Context Clues paper",
    url="https://github.com/som-shahlab/long_context_clues/",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
)