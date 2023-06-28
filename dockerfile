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

# Updates and packages
RUN apt-get update && \
    apt-get -yq dist-upgrade && \
    apt-get install -yq apt-utils software-properties-common locales && \
    apt-get install -yq git python3 python3-pip && \
    apt-get install -yq dos2unix ca-certificates nano

# User management
RUN adduser --quiet --disabled-password --shell /bin/bash --gecos "Kari" kamms && \
    echo "kamms:kamms" | chpasswd

RUN mkdir /proteogyver
COPY docker_entrypoint.sh /
COPY app/update.sh /update.sh
COPY app /proteogyver
WORKDIR /proteogyver

# Python installs
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install setuptools gunicorn


# Install and setup R
RUN apt-get install -y \
    software-properties-common dirmngr wget
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get update && \
    apt-get install -y \
    littler \
    r-cran-littler \
    r-base \
    r-base-dev \
    r-recommended
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> /etc/R/Rprofile.site
WORKDIR /proteogyver/resources
RUN Rscript R_requirements.R

WORKDIR /proteogyver/external/SAINTexpress
RUN chmod 777 SAINTexpress-spc
RUN chmod 777 SAINTexpress-int
WORKDIR /

RUN apt-get -yq install postgresql

# JupyterHub
RUN apt-get -yq install npm nodejs && \
    npm install -g configurable-http-proxy
RUN pip3 install jupyter jupyterlab jupyterhub pandas jupyter-dash
RUN mkdir /etc/jupyterhub
COPY jupyterhub.py /etc/jupyterhub/

COPY additional/data.tar /proteogyver/data.tar
COPY additional/external.tar /proteogyver/external.tar
WORKDIR /proteogyver
RUN tar xf data.tar
RUN tar xf external.tar

# Expose ports (jupyterHub. dash)
EXPOSE 8090 8050

# Finished.
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]