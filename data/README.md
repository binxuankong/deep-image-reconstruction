# Deep Image Reconstruction from Human Brain Activity
## Data

The data required to run the program should be saved in the respective directories as follows.

* *eeg* - processed EEG data of 5 subjects
* *eeg_decoded* - decoded hidden representation of image features from EEG data
* *encoded_features* - hidden representation of image features
* *fmri* - processed fMRI data of 5 subjects
* *fmri_decoded* - decoded hidden representation of image features from fMRI data
* *images* - images used in the thesis (18 categories)
* *img_features* - image features extracted from pre-trained VGG19 (all 19 layers)
* *trained_models* - trained autoencoder for encoding and decoding of image features
* *TRAINED_TEST.p* - dictionary containg the IDs of training and testing images

Unfortunately the data used for this project is not available publicly. The code should be able to run with different datasets with slight modification as long as they are in the right format.
