FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN echo "export http_proxy=http://proxytc.vingroup.net:9090" >> /etc/environment
RUN echo "export https_proxy=http://proxytc.vingroup.net:9090" >> /etc/environment
RUN echo "Acquire::http::proxy \"http://proxytc.vingroup.net:9090\";" >> /etc/apt/apt.conf.d/proxy.conf
RUN echo "Acquire::https::proxy \"http://proxytc.vingroup.net:9090\";" >> /etc/apt/apt.conf.d/proxy.conf

RUN pip3 install beautifulsoup4
RUN pip3 install lxml
RUN pip3 install networkx
RUN pip3 install nltk
RUN pip3 install numpy
RUN pip3 install optuna
RUN pip3 install pandas
RUN pip3 install rouge
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install spacy
# RUN pip3 install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install adapter-transformers
# RUN pip3 install pytorch-lightning
RUN pip3 install transformers
RUN pip3 install sentence-transformers
RUN pip3 install sentencepiece

# docker build -t hieumdt/ie_env -f information-extraction-env.dockerfile .
