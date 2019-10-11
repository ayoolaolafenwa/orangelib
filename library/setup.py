# EDIT THE setup.py file

#This file provides the information needed to build our library.
#Put the following code into it.
#You can edit the version, description, url and author as needed.


from setuptools import setup,find_packages

setup(name="orangelib",
      version='0.1.0',
      description='A library for classifying oranges',
      url="https://github.com/ayoolaolafenwa/orangelib",
      author='Ayoola Olafenwa',
      license='MIT',
      packages= find_packages(),
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
