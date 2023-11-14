

FROM pgbase:pgbase
LABEL maintainer "Kari Salokas kari.salokas@helsinki.fi"

# First steps
USER root
WORKDIR /
RUN mkdir /proteogyver
COPY docker_entrypoint.sh /
COPY app/update.sh /update.sh
COPY app /proteogyver

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

# Unpack database
WORKDIR /proteogyver/data
RUN unxz proteogyver.db.xz
RUN mkdir -p /etc/supervisor/conf.d
RUN cp /proteogyver/other_commands/celery.conf /etc/supervisor/conf.d/celery.conf
# Python installs
WORKDIR /proteogyver
RUN sed -i 's\"/mnt", "c", "DATA", "PG cache"\"/proteogyver", "cache"\g' parameters.json  
RUN mkdir /proteogyver/cache
WORKDIR /proteogyver/resources
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed -r requirements.txt

# Expose ports (jupyterHub. dash)
EXPOSE 8090 8050

# Finished.
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]