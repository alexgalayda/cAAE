FROM nvcr.io/nvidia/pytorch:19.12-py3

ARG TRAIN
ARG CONFIG_NAME
ENV CONFIG_NAME=$CONFIG_NAME
WORKDIR /root

#jupyter notebook
RUN jupyter notebook --generate-config
COPY config/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

COPY $TRAIN/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

ADD model /root/model
ADD config /root/config

WORKDIR /root
CMD python3 ./model/train.py -c $CONFIG_NAME