from setuptools import setup, find_packages

setup(
    name="franka-sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Baris Yazici",
    author_email="barisyazici@alumni.sabanciuniv.edu",
    description="A Franka robot simulation server",
    keywords="robotics,simulation,franka",
    python_requires='>=3.6',
) 