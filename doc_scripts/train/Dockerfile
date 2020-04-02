FROM nvcr.io/nvidia/pytorch:19.12-py3

ARG TRAIN

WORKDIR /root

#jupyter notebook
RUN jupyter notebook --generate-config
COPY config/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

COPY $TRAIN/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt




#ADD ./train/model /root/model/
#ADD model /root/model


#RUN echo TEST:${HCP}
#ENV {}
#RUN mkdir HCP
#cp -r lol/* shara/ && cd shara
#python3 main.py --model_name "cAAE" --z_dim "128"

#CMD python3 train.py


WORKDIR /root
#CMD jupyter notebook
CMD python3 ./model/train.py -c $TRAIN