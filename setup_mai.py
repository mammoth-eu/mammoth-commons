import setuptools

# Developer self-reminder for uploading to pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - build  : python setup_mai.py bdist_wheel
# - deploy : twine upload dist/*
# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server

with open("README.md", "r") as file:
    long_description = "This is the desktop version of the MAI-BIAS toolkit.\nFor more information visit the [homepage](https://github.com/mammoth-eu/mammoth-commons)."

with open("config/requirements[mai].txt", "r") as file:
    deployment_requirements = file.read().splitlines()

setuptools.setup(
    name="MAI-Bias",
    version="0.2.35",
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Desktop app version of the MAI-Bias toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mammoth-eu/mammoth-commons",
    packages=setuptools.find_packages(),
    package_data={
        "mai_bias.icons": ["*.png", "*.svg", "*.jpg", "*.jpeg", "*.gif"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=deployment_requirements,
)
