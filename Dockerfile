FROM continuumio/miniconda3
RUN conda install mpi4py 
COPY requirements.txt /tmp/
RUN conda config --add channels conda-forge
RUN conda install --file /tmp/requirements.txt
RUN conda install -c conda-forge treecorr
RUN pip install chainconsumer twopoint 
RUN git clone https://github.com/des-science/destest.git && cd destest && git fetch && git checkout setup && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN git clone https://github.com/des-science/2pt_pipeline.git && cd 2pt_pipeline && git fetch && git checkout python3

ENV XDG_CACHE_HOME=/srv/cache
RUN mkdir -p $XDG_CACHE_HOME/astropy
ENV XDG_CONFIG_HOME=/srv/config
RUN mkdir -p $XDG_CONFIG_HOME/astropy
RUN python -c "import astropy"
RUN /sbin/ldconfig
