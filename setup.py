"""Setup script for MovAnal package."""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="movAnal",
    version="0.1.0",
    description="Facility Anomaly Detection System using Video Analysis",
    author="MovAnal Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "movAnal-train=scripts.train:main",
            "movAnal-inference=scripts.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
