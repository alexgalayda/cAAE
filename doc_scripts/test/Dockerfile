FROM nvcr.io/nvidia/pytorch:19.12-py3

ARG TEST
ARG CONFIG_NAME
ENV CONFIG_NAME=$CONFIG_NAME
WORKDIR /root

COPY $TEST/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

WORKDIR /root
ADD model /root/model
ADD config /root/config

WORKDIR /root
CMD python3 ./model/test.py -c $CONFIG_NAME