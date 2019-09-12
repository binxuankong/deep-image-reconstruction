# Deep Image Reconstruction from Human Brain Activity
## Brain Activity Decoding

This file contains the script to perform feature decoding from brain activity (for both fMRI and EEG data).

We used the same methodology as [Horikawa & Kamitani, 2017, Generic decoding of seen and imagined objects using hierarchical visual features, Nat Commun.](https://www.nature.com/articles/ncomms15037). Due to the lack of computational power, we use autoencoders to encode extracted images for each layer into a smaller dimension and perform the decoding on the hidden representation instead of the whole features. The feature decoding scripts (*feature_predict.py*, *feature_prediction_fmri.py*, and *feature_prediction_eeg.py*) are based on their demo programs for Python, available at https://github.com/KamitaniLab/GenericObjectDecoding.

### Usage

1. *extract_img_features.py* - extract the features of the images using pre-trained VGG19 for each layer
2. *normalize_features.py* - normalize the image features by saving the mean and standard deviation for each layer
3. *train_autoencoder.py* - train autoencoders on image features
4. *encode_features.py* - encode the image features using trained encoders
5. *feature_prediction_fmri.py* & *feature_prediction_eeg.py* - decode encoded features from fMRI or EEG