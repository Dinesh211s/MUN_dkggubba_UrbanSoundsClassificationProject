# MUN_dkggubba_UrbanSoundsClassificationProject
Urban sounds classification using machine learning classification algorithms such as Random forest, Bagging classifier, Gradient Boosting, KNN. 
Main task is to find out which algorithm is mainly producing better results and less error rate.

Dataset:
 A total of 8732 samples (*=4s) were extracted from this dataset. 
 They are categorized into 10 categories: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine idling, gun shot, jackhammer, siren, and street music.
 The classes are based on the urban sound taxonomy. The excerpts were taken from recordings uploaded to www.freesound.org.
 In order to make comparisons with the automatic classification results described in the article above, files are sorted into ten folds (folders numbered fold1-fold10).
 A CSV file containing metadata about each excerpt is included with the sound excerpts. 
 
 Audio files Included:
 We have 8732 audio files of urban sounds in WAV format.The sampling rate, bit depth, 
 and number of channels are the same as that of the original file uploaded to Freesound.
 
Meta data files:
Urbansoundscalssification.csv Download the full dataset from https://urbansounddataset.weebly.com/download-urbansound8k.html

The Urbansounds8K dataset contains ten folds of audio samples with a length of (<4s). 
Slice file name: contains names of audio files.
Fsid : freesound id of the audio file
Class id: As there are ten classes of sounds it represents which class represents the given audio file.
Start: start time of the slice in the original free sound recording
End: represents the end slice in the original free sound recording
Salience: 1- background, 2- foreground
Fold: There are 10 folders in which folder that particular audio file contains.
Class: car_horn,airconditioner,childrenplaying,dogbark,drilling,jackhammer,engineidling,gunshot,streetmusic,siren.

 
 
