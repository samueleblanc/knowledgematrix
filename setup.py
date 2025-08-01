from setuptools import setup, find_packages

setup(
    name="knowledgematrix",
    version="0.0.1",
    description="Compute the knowledge matrices",
    author="Marco Armenta and Samuel Leblanc",
    maintainer="Samuel Leblanc",
    url="https://github.com/samueleblanc/knowledgematrix",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)