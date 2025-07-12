from setuptools import setup, find_packages

setup(
    name="sgmt",
    version="0.1.0",
    author="Jasraj Budigam",
    author_email="jasraj@example.com",
    description="Slot-Guided Modular Transformer for compositional tasks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/sgmt",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7",
        "einops",
        "sentencepiece",
        "tqdm",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "nbformat",
        "dill"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.8',
    license="MIT",
)
