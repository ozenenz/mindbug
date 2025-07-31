from setuptools import find_packages, setup

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mindbug-deep-cfr",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep CFR implementation for Mindbug First Contact",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mindbug-deep-cfr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "tensorboard>=2.8.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-mindbug=mindbug.training.train:main",
            "train-mindbug-advanced=mindbug.training.train_improved:main",
        ],
    },
)