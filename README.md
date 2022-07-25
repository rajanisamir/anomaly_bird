# anomaly_bird
Self-Supervised Anomaly Detection on the BirdAudio Dataset at Argonne National Laboratory

## Goal
The purpose of this repository is to detect anomalies in a set of audio files recorded by [Waggle sensors](https://github.com/waggle-sensor/waggle) at the Morton Arboretum in Lisle, IL, near Argonne National Laboratory. This is accomplished using a custom convolutional autoencoder. During evaluation, out-of-distribution embeddings and samples with large reconstruction errors are likely anomalies. Currently, I'm working on a technique to prune anomalies during training to improve the use of the autoencoder for classification of bird sounds in the recording.

## File Structure
- The directories `theta_run_birds` and `theta_run_noise` currently house Tensorboard logs and model checkpoints for tests that I have run on ThetaGPU nodes at the Argonne Leadership Computing Facility. 
- `ae_model.py` contains the architecture of the custom autoencoder. The encoder consists of four convolutional layers followed by ReLU activation and max-pooling. A linear layer and a softmax produce the embedding at the bottleneck, and the architecture of the decoder is symmetric.
- `ae_recon.ipynb` houses the evaluation code, in which k-means is used to cluster the embeddings at the bottleneck, and PCA is used to reduce them to 2 or 3 dimensions for visualization. An interactive plot is made with [Plotly](https://plotly.com/python/), where hovering over points shows their timestamps in the original audio file.
- `ae_settings.py` contains some basic settings for the autoencoder training, and is simply imported in `ae_train.py`
- `ae_train.py` contains the bulk of the code for training the autoencoder. It also contains experimental code for pruning anomalies during training.
- `custom_audio_dataset.py` creates a PyTorch Dataset based on a list of audio files by preprocessing the data (e.g. mixing it down to one channel, cropping the spectrograms, etc.). The code in this file was originally adapted from Valerio Velarado's [excellent video series](https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm) of processing audio data on PyTorch, with several features added by me.
- `distributed.py` was taken from the repository for [VICReg](https://github.com/facebookresearch/vicreg), which contains a PyTorch implementation of a novel self-supervised learning technique. It contains functions that help set up distributed parallel training.
- `train.sh` is a basic bash script used to submit a training run to ThetaGPU at ALCF.
