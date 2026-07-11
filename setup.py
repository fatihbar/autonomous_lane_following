from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="autonomous_lane_following",
    version="0.1.0",
    author="Fatih Barlas",
    author_email="94764881+fatihbar@users.noreply.github.com",
    description="Professional autonomous lane following system with safety features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fatihbar/autonomous_lane_following",
    project_urls={
        "Bug Tracker": "https://github.com/fatihbar/autonomous_lane_following/issues",
        "Documentation": "https://github.com/fatihbar/autonomous_lane_following/tree/main/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "scikit-image>=0.21.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "ros2": [
            "rclpy>=3.0.0",
            "sensor-msgs>=0.1.0",
            "geometry-msgs>=0.1.0",
            "cv-bridge>=3.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "pylint>=2.17.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lane-follower=scripts.run_standalone:main",
            "test-detectors=scripts.test_detectors:main",
        ],
    },
    include_package_data=True,
)
