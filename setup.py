from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autonomous_lane_following",
    version="0.1.0",
    author="Autonomous Systems Team",
    description="Professional autonomous lane following system for closed-track vehicles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fatihbar/autonomous_lane_following",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
