"""
Setup script for the compression pipeline package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="compression-pipeline",
    version="1.0.0",
    author="Compression Pipeline Team",
    author_email="team@compressionpipeline.com",
    description="A comprehensive data compression pipeline using DCT, quantization, and Huffman encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/compression-pipeline/compression-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Archiving :: Compression",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "matplotlib>=3.0",
            "jupyter>=1.0",
        ],
        "examples": [
            "matplotlib>=3.0",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "compression-benchmark=examples.performance_benchmarks:main",
            "compression-demo=examples.basic_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "compression_pipeline": ["*.md"],
        "examples": ["*.py"],
        "tests": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/compression-pipeline/compression-pipeline/issues",
        "Source": "https://github.com/compression-pipeline/compression-pipeline",
        "Documentation": "https://compression-pipeline.readthedocs.io/",
    },
)