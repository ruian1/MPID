# MPID(PyTorch): Multiple Particle Identification, PyTorch version

PyTorch version for MPID. Rewritten from the TensorFlow version for the purpose of production on Fermilab GRID
Multiple Particle Identification CNN. MPID taks a 512x512 LArTPC images and returns probablities of having particles from proton, electron, muon, muon and charged pions.

# Dependecies:
[LArCV](https://github.com/LArbys/LArCV),
ROOT,
PyTorch

# Setup:
0. Setup LArCV using [Wiki](https://github.com/LArbys/LArCV),
1. git clone https://github.com/ruian1/MPID_pytorch.git
2. source setup.sh

# Training:
0. Edit training configures in under ./uboone/run_train.sh
1. source ./uboone/run_train.sh

# Inference:
0. cd ./uboone
1. python mpid_inference.py	image2d.root (for [image2d](https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/Image2D.h) input).
2. python inference_pid_torch.py	pixel2d.root (for [pixel2d](https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/Pixel2D.h) input).
