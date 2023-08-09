"""
    setup script for the API
"""
from setuptools import find_packages
from setuptools import setup

with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()

requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="argo_ml",
    version="1.0",
    description="Project Description",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
