# Seizure-prediction-CNN

Command to run the code: </br>
python main.py --mode MODE --dataset DATASET

MODE: cv, test

-- cv: leave-one-out cross-validation </br>
-- test: ~1/3 of last seizures used for test, interictal signals are split accordingly

DATASET: CHBMIT, FB, Kaggle2014Pred

Requirements: </br>
h5py (2.7.1) </br>
hickle (2.1.0) </br>
Keras (2.0.6) </br>
matplotlib (1.3.1) </br>
mne (0.11.0) </br>
pandas (0.21.0) </br>
scikit-learn (0.19.1) </br>
scipy (1.0.0) </br>
tensorflow-gpu (1.4.1)
