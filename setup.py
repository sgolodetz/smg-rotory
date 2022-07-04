from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smg-rotory",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Interfaces for controlling various drones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-rotory",
    packages=find_packages(include=["smg.rotory", "smg.rotory.*"]),
    include_package_data=True,
    install_requires=[
        "av",
        "netifaces",
        "numpy",
        "opencv-contrib-python==3.4.2.16",
        "smg-imagesources",
        "smg-rigging"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
