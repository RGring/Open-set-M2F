From detectron2:latest

# Install dependencies
COPY requirements.txt /open-set-m2f/requirements.txt
RUN pip install -r /open-set-m2f/requirements.txt

COPY mask2former /open-set-m2f/mask2former
WORKDIR /open-set-m2f/mask2former/modeling/pixel_decoder/ops
RUN ./make.sh

ENV DETECTRON2_DATASETS /data
WORKDIR /open-set-m2f
CMD ["/bin/bash"]