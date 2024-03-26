### Software Development /Object Oriented Programming (OOP) Practices:
I'm a big advocate of writing re-usable and streamline code for data science projects. Some of the modules that I've built for my personal bioinformatics projects can be found on [OmixHub](https://github.com/adhal007/OmixHub). Some of the crucial and advanced concepts of **inheritance, polymorphism, encapsulation and abstraction** can be seen in all of the modules.

Here is a list of some useful tools that can be found:

- **[Base Preprocessor](https://github.com/adhal007/OmixHub/blob/main/src/base_preprocessor.py)**
  - Example class with methods and attributes inherited by child classes. Some key methods provided are:
    - data skew
    - data leakage
    - patient overlap
    - training_testing_split 
    - etc
- **[Base ml models wrapper](https://github.com/adhal007/OmixHub/blob/main/src/base_ml_models.py)**
  - Example class with functionality to evaluate and plot multiple ML models for a data science application. This is intended to be inherited by specific child classes for building models for different Omics data
- **[Dimensionality reduction and clustering wrapper](https://github.com/adhal007/OmixHub/blob/main/src/DimRedMappers/README.md)**
  - Interfaces with Preprocessor classes to facilitate easy application of UMAP and clustering
- **[Differential Analysis Wrapper](https://github.com/adhal007/OmixHub/blob/main/src/pydeseq_utils.py)**
  - Faciliatates easy application of pydyseq in a few lines to perform differential analysis