# Neural-mask-estimation

# key feature

- LSTM-based Neural Mask Estimation for designing MVDR [1, 4]
- on-the-fly data augmentation
- pre-trained model	
- speaker-Aware mask training supported [2]
- SNR-based reference mic selection for MVDR [1, 4]
- small scale sample training data 
	- You can perform experiment using any data by replacing the data
	- We put WHAM! noise data[2], Libri Speech and LJ speech as sample noise clean speech data.


# How to use

1. Please run generate_validate_data.py
	- Please put data(noise and clean speech) ./dataset/validate/*
	- You will get validation_features/speech_mask.npy, validation_features/noise_mask.npy and validation_features/val_spec.npy

2. Please run train.py
	- Please put data(noise and clean speech) ./dataset/train/*
	- You will get model/neaural_mask_estimator{}.hdf5
		・{} indicates the number of times of epoch	

3. Please run predict.py
	- Perform mask estimation and design MVDR beamformer and you can get enhanced speech
	- Please put multi channel data ./dataset/data_for_beamforming/* for beamforming
	- You will get result in ./result/*
		・ enhencement_all_channels.wav is result without channel selection
		-・enhacement_snr_select.wav is result with channel selection
	

speaker-aware mask estimating

1: Please run adapt.py
	- Please prepare target speaker list and non target speaker list (e.g., sp1_list.txt, sp2_list.txt)
	- you will get speaker-aware model ./model/speaker_2.hdf5	
	
2. Please run speaker_aware_mask_predict.py
	- you can compare mask results before/after adaptation 
		
# Reference:

	[1] EXPLORING PRACTICAL ASPECTS OF NEURAL MASK-BASED BEAMFORMING FOR FAR-FIELD SPEECH RECOGNITION
		- https://www.microsoft.com/en-us/research/uploads/prod/2018/04/ICASSP2018-Christoph.pdf


	[2] WHAM!: Extending Speech Separation to Noisy Environments
		- https://arxiv.org/abs/1907.01160
		
	[3] The Hitachi/JHU CHiME-5 system: Advances in speech recognition for veryday home environments using multiple microphone arrays
		- http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_kanda.pdf
	
	
	[4] Improved MVDR beamforming using single-channel mask prediction networks
		- https://www.merl.com/publications/docs/TR2016-072.pdf
    
![sample_mask](https://user-images.githubusercontent.com/41845296/62979654-5b090880-be5f-11e9-8eb3-08afc616e279.png)
![sample_mask_multi](https://user-images.githubusercontent.com/41845296/62979655-5b090880-be5f-11e9-9fde-028cc82d4f33.png)
![model](https://user-images.githubusercontent.com/41845296/62979656-5b090880-be5f-11e9-9e4f-fa4e4be17560.png)

# Requirement:
python                    3.6.7+

numpy	                  1.14.3
soundfile                 0.9.0
pyroomacoustics           0.1.21
librosa                   0.6.2
tensorflow                1.9.0
scipy                     1.2.0
cython                    0.25.2
matplotlib                3.6.7


    
    
