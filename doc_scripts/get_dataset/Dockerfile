FROM nvcr.io/nvidia/pytorch:19.12-py3

ARG GET_DATASET

WORKDIR /root
COPY $GET_DATASET/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

COPY $GET_DATASET/copy_dataset.py copy_dataset.py

COPY $GET_DATASET/credentials .aws/credentials

CMD python3 copy_dataset.py