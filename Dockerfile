FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y autoconf automake gcc gfortran g++ zlib1g-dev make cmake wget make git swig pkg-config python3 python-pip python3-pip libffi-dev libssl-dev libgsl-dev libfftw3-dev && apt-get clean all

RUN mkdir /build/
#RUN cd /build && wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz \
#  && tar xvzf Python-3.6.1.tgz && cd /build/Python-3.6.1 \
#  && ./configure && make -j4 && make install && make clean && rm /build/Python-3.6.1.tgz
RUN cd /build && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
  && tar xvzf mpich-3.2.tar.gz && cd /build/mpich-3.2 \
  && ./configure && make -j4 && make install && make clean && rm /build/mpich-3.2.tar.gz
RUN cd /build && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz \
  && tar xvzf mpi4py-2.0.0.tar.gz
RUN cd /build/mpi4py-2.0.0 && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN cat /tmp/requirements.txt | xargs -n 1 -L 1 pip3 install
ARG CSBUST=0
RUN git clone https://github.com/des-science/destest.git && cd destest && git fetch && git checkout setup && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN git clone https://github.com/des-science/2pt_pipeline.git && cd 2pt_pipeline && git fetch && git checkout python3 

ENV XDG_CACHE_HOME=/srv/cache
RUN mkdir -p $XDG_CACHE_HOME/astropy
ENV XDG_CONFIG_HOME=/srv/config
RUN mkdir -p $XDG_CONFIG_HOME/astropy
RUN python -c "import astropy"
RUN /sbin/ldconfig
