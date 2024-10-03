FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND=noninteractive

RUN apt purge python3.6\* -y

RUN apt-get update && apt-get install -y \
	python3.8 python3-opencv ca-certificates python3.8-dev python3.8-distutils git wget sudo ninja-build

RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN ln -sv /usr/bin/python3 /usr/bin/python


# ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

RUN python3 -m pip install --upgrade pip

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip uninstall numpy -y 
RUN pip install opencv-python
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install setuptools==59.5.0
RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install git+https://github.com/mcordts/cityscapesScripts.git
RUN pip install -e detectron2
