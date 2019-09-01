# Convolutional neural networks for seizure prediction using intracranial and  scalp  electroencephalogram

##### Code to reproduce results reported in our paper published as:
Truong, N. D., A. D. Nguyen, L. Kuhlmann, M. R. Bonyadi, J. Yang, S. Ippolito, and O. Kavehei (2018). "Convolutional neural networks for seizure prediction using intracranial and  scalp  electroencephalogram." *Neural  Networks* 105, 104-111. DOI:10.1016/j.neunet.2018.04.018.

#### Requirements
* h5py (2.7.1)
* hickle (2.1.0)
* Keras (2.0.6)
* matplotlib (1.3.1)
* mne (0.11.0)
* pandas (0.21.0)
* scikit-learn (0.19.1)
* scipy (1.0.0)
* tensorflow-gpu (1.4.1)

#### How to run the code
1. Set the paths in \*.json files. Copy files in folder "copy-to-CHBMIT-folder" to your CHBMIT dataset folder.

2. Run the code
```console
python main.py --mode MODE --dataset DATASET
```
##### where: </br>
* MODE: cv, test
  * cv: leave-one-seizure-out cross-validation
  * test: ~1/3 of last seizures used for test, interictal signals are split accordingly
* DATASET: FB, CHBMIT, Kaggle2014Pred
