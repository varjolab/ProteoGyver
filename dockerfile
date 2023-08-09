

FROM ubuntu:22.04
LABEL maintainer "Kari Salokas kari.salokas@helsinki.fi"

ENV DEBIAN_FRONTEND noninteractive
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV R_BASE_VERSION 3.6.1

# First steps
USER root
WORKDIR /
RUN mkdir /proteogyver
COPY docker_entrypoint.sh /
COPY app/update.sh /update.sh
COPY app /proteogyver

# Install and setup R and other software
# Updates and packages
RUN apt-get update && \
    apt-get -yq dist-upgrade && \
    apt-get install -yq \
    software-properties-common dirmngr wget
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

RUN apt-get install -yq apt-utils software-properties-common locales \
    git python3 python3-pip nodejs npm  \
    dos2unix ca-certificates nano  \
    littler \
    r-cran-littler \
    r-base \
    r-base-dev \
    r-recommended 


# Python installs
WORKDIR /proteogyver/resources
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter jupyterlab jupyterhub pandas jupyter-dash gunicorn
# R things
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site
#RUN Rscript R_requirements.R
RUN npm install -g configurable-http-proxy

# User management
RUN adduser --quiet --disabled-password --shell /bin/bash --gecos "Kari" kamms && \
    echo "kamms:kamms" | chpasswd

# Make SAINT executable
WORKDIR /proteogyver/external/SAINTexpress
RUN chmod 777 SAINTexpress-spc
RUN chmod 777 SAINTexpress-int
WORKDIR /
# JupyterHub
RUN mkdir /etc/jupyterhub
COPY jupyterhub.py /etc/jupyterhub/

# Put additional data where it should be
COPY additional/data.tar /proteogyver/data.tar
COPY additional/external.tar /proteogyver/external.tar
WORKDIR /proteogyver
RUN tar xf data.tar
RUN tar xf external.tar
RUN mkdir /proteogyver/debug


# Expose ports (jupyterHub. dash)
EXPOSE 8090 8050

# Finished.
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]