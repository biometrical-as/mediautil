import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mediautil",
    version="0.2",
    author="Martin Jordal Hovin",
    author_email="martin@biometrical.io",
    description="Quality of life package for using images and video in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/biometrical-as/mediautil",
    packages=["mediautil"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=["bin/mplayer"],
    entry_points={"console_scripts": ["mediaplayer = mediautil.run:app"]},
    python_requires=">=3.7",
)
