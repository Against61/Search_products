# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]





# defune package
setup(
    name="im_retrieval",
    version=0.1,
    description="Image Retrieval task, inference and populate elastic search DB.",
    author="Against61",
    author_email="against61@gmail.com",
    python_requires=">=3.9",
    install_requires=[required_packages],
)
