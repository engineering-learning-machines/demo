#!/bin/bash
#
export DEBIAN_FRONTEND=noninteractive

#
sudo apt-get update

#
sudo apt-get install -y python3-pip

#
sudo apt-get install -y xorg-dev

#
sudo apt-get install -y python3-tk

#
sudo apt-get install unzip

#
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/390.48/NVIDIA-Linux-x86_64-390.48.run

#
chmod +x NVIDIA-Linux-x86_64-390.48.run

#
sudo ./NVIDIA-Linux-x86_64-390.48.run  --accept-license --no-opengl-files --silent

#
pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl

#
sudo pip3 install -y torchvision

#
sudo pip3 install -y fastai
