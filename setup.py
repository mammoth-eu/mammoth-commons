import setuptools

# Developer self-reminder for uploading to pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*
# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server


with open("README.md", "r") as file:
    long_description = "Quickly develop modules for the MAI-BIAS toolkit.\nFor more information visit the [homepage](https://github.com/mammoth-eu/mammoth-commons)."

with open("requirements.txt", "r") as file:
    deployment_requirements = file.read().splitlines()

setuptools.setup(
    name="MAMMOth-commons",
    version="0.1.35",
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Component interfaces of the MAI-BIAS toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mammoth-eu/mammoth-commons",
    packages=[
        package
        for package in setuptools.find_packages()
        if package.startswith("mammoth_commons")
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=deployment_requirements,
    extras_require={"deployment": deployment_requirements},
)
