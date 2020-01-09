from setuptools import setup,find_packages

setup(name="orangelib",
      version='0.4.0',
      description='Orangelib is a library for classifying fruits. It classifies the following categories of fruits:- Ripe and Unripe Oranges: Ripe and Unripe Bananas: Green and Red Apples',
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
