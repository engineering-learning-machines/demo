#!/bin/bash
#
export DEBIAN_FRONTEND=noninteractive

#
sudo apt-get install -y xorg-dev

#
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/390.48/NVIDIA-Linux-x86_64-390.48.run

#
chmod +x NVIDIA-Linux-x86_64-390.48.run

#
sudo ./NVIDIA-Linux-x86_64-390.48.run  --accept-license --no-opengl-files --silent

#
sudo apt-get install -y python3-pip

#
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl 

#
pip3 install torchvision

#
pip3 install fastai
