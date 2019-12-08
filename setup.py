

#This file provides the information needed to build our library.



from setuptools import setup,find_packages

setup(name="orangelib",
      version='0.2.0',
      description='A library for classifying oranges into two classes:ripe and unripe ',
      url="https://github.com/ayoolaolafenwa/Oranges_classifier",
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
