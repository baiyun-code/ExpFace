# Install the developing environment for arcface_pytorch on Ubuntu

As it is required to use MXNet while reading data, we tried to install the environment for arcface_pytorch. However, we failed to install it on Windows. Fortunately, we successfully installed it on Ubuntu 22.04. Although there are many tricks and no guarantee of 100% success, we write a document for instructing the complex deploying process.

1. Install Anaconda as the environment to manage all packages.
2. Create one virtual environment to allow failures.
   - `conda create -n arcface python==3.11`
   - We need a lower version of python in order to match the lower version of numpy.
3. Activate the environment
   - `conda activate arcface`
4. Install pytorch. This one is easy without tricks.
   - `conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
5. Install numpy.
   - `pip install numpy==1.23.5`
   - Mxnet uses the function bool() in numpy. However, this function is disabled in the version of 1.24.
6. Install mxnet
   - `pip install mxnet-cu117`
   - It seems no cu117 version on windows. Thus, we failed to install it on windows.
   - The mxnet-cu117 requires nccl. Thus, we need to install cudnn and nccl.
7. Install cudnn+cudatoolkit
   - `conda install cudnn==8.9.2.26`
   - The cudatoolkit will be installed automatically.
8. Install nccl.
   - Please type the following commands one by one.
       - wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
       - sudo dpkg -i cuda-keyring_1.0-1_all.deb
       - sudo apt-get update
       - sudo apt install libnccl2 libnccl-dev
   - We are not 100% confident with the method above. Please refer to the link below if failed. Registration is needed.
       - https://developer.nvidia.com/nccl/nccl-legacy-downloads
9. Install tensorboard
    - `pip install tensorboard`
10. Install scikit-learn
    - `pip install scikit-learn`
11. Install easydict
    - `install easydict`