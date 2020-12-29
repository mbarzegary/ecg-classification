# ECG Signals Classification

This repository contains code to classify ECG signals to detect seizure by labeling beats according to the AAMI recommendation. The code has been used in the following papers (in which you can find the detail of the classification process), so please cite these papers if you wish to use this code in your research.

    @article{Farahani2020,
    title = "Towards collaborative intelligent IoT eHealth: From device to fog, and cloud",
    journal = "Microprocessors and Microsystems",
    volume = "72",
    pages = "102938",
    year = "2020",
    issn = "0141-9331",
    doi = "https://doi.org/10.1016/j.micpro.2019.102938",
    url = "http://www.sciencedirect.com/science/article/pii/S0141933119303928",
    author = "Bahar Farahani and Mojtaba Barzegari and Fereidoon {Shams Aliee} and Khaja Ahmad Shaik",
    }

    @inproceedings{Farahani2019,
    author = {Farahani, Bahar and Barzegari, Mojtaba and Aliee, Fereidoon Shams},
    title = {Towards Collaborative Machine Learning Driven Healthcare Internet of Things},
    year = {2019},
    isbn = {9781450366403},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3312614.3312644},
    doi = {10.1145/3312614.3312644},
    booktitle = {Proceedings of the International Conference on Omni-Layer Intelligent Systems},
    pages = {134–140},
    numpages = {7},
    keywords = {Machine Learning, Health, Internet of Things},
    location = {Crete, Greece},
    series = {COINS '19}
    }

The code is originally based on the [Mondejar's work](https://github.com/mondejar/ecg-classification), but it contains several improvements such as porting to Python 3, bug fixes, and performance improvements. It also contains more models in comparison to the original work.

## Getting started

After cloning the repository, you should first download the dataset. The code is developed and tested based on the MIT-BIH database, which can be downloaded from [Kaggle website](https://www.kaggle.com/mondejar/mitbih-database). Then, the main entry point of the code, `run_train_SVM.py` can be run. The code uses scikit-learn, so a typical setup of Numpy, matplotlib, and scikit-learn is required to be installed in the Python environment.
