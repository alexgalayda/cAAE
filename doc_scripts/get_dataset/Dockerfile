FROM nvcr.io/nvidia/tensorflow:19.10-py3

WORKDIR /root
COPY ./get_dataset/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
COPY ./get_dataset/copy_dataset.py copy_dataset.py
RUN mkdir .aws
COPY ./get_dataset/credentials .aws/
#CMD bash
CMD ["python3", "copy_dataset.py"]
