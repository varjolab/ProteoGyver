

FROM pgbase:1.1
LABEL maintainer "Kari Salokas kari.salokas@helsinki.fi"

# First steps
USER root
WORKDIR /
# Create mounts for the data
#RUN mkdir /proteogyver/data
RUN mkdir /proteogyver/data/Server_output
RUN mkdir /proteogyver/data/MS_rundata
COPY docker_entrypoint.sh /
COPY app/update.sh /update.sh
COPY app /proteogyver
COPY docker_entrypoint.sh /docker_entrypoint.sh
# Make SAINT executable
WORKDIR /proteogyver/external/SAINTexpress
RUN chmod 777 SAINTexpress-spc
RUN chmod 777 SAINTexpress-int
RUN ln -s /proteogyver/external/SAINTexpress/SAINTexpress-spc /usr/bin/SAINTexpressSpc
RUN ln -s /proteogyver/external/SAINTexpress/SAINTexpress-int /usr/bin/SAINTexpressInt

WORKDIR /
# JupyterHub
RUN mkdir /etc/jupyterhub
COPY jupyterhub.py /etc/jupyterhub/

# Unpack database
WORKDIR /proteogyver/data
RUN unxz proteogyver.db.xz
RUN mkdir -p /etc/supervisor/conf.d
RUN cp /proteogyver/other_commands/celery.conf /etc/supervisor/conf.d/celery.conf
# Python installs in case requirements got updated since pg_base was built.
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed -r requirements.txt
WORKDIR /proteogyver
RUN sed -i 's\"/home", "kmsaloka", "Documents", "PG_cache"\"/proteogyver", "cache"\g' parameters.json  
RUN sed -i 's\"Local debug": true\"Local debug": false\g' parameters.json  
RUN mkdir /proteogyver/cache
WORKDIR /proteogyver/resources

# Expose ports (jupyterHub. dash)
EXPOSE 8090 8050

# Finished.
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]