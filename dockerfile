FROM pgbase:1.2
LABEL maintainer "Kari Salokas kari.salokas@helsinki.fi"

# First steps
USER root

RUN mkdir -p /proteogyver/data/Server_output
RUN mkdir -p /proteogyver/data/MS_rundata
RUN mkdir -p /etc/supervisor/conf.d
RUN mkdir -p /etc/jupyterhub
RUN mkdir -p /proteogyver/cache
RUN mkdir -p /proteogyver/data/unparsed_stats
RUN mkdir -p /proteogyver/data/Server_output/stats

WORKDIR /
# Create mounts for the data
COPY docker_entrypoint.sh /
COPY app/update.sh /update.sh
COPY app /proteogyver
COPY docker_entrypoint.sh /docker_entrypoint.sh
COPY jupyterhub.py /etc/jupyterhub/
COPY nm_pack.py /nm_pack.py
# Make SAINT executable
WORKDIR /proteogyver/external/SAINTexpress
RUN chmod 777 SAINTexpress-spc
RUN chmod 777 SAINTexpress-int
RUN ln -s /proteogyver/external/SAINTexpress/SAINTexpress-spc /usr/bin/SAINTexpressSpc
RUN ln -s /proteogyver/external/SAINTexpress/SAINTexpress-int /usr/bin/SAINTexpressInt

COPY app/Utilities/cron_maintenance_jobs /etc/cron.d/cron_maintenance_jobs
RUN chmod 0644 /etc/cron.d/cron_maintenance_jobs
RUN crontab /etc/cron.d/cron_maintenance_jobs
RUN touch /var/log/cron.log

# JupyterHub

# Unpack database
WORKDIR /proteogyver/data
RUN unxz proteogyver.db.xz
RUN cp /proteogyver/other_commands/celery.conf /etc/supervisor/conf.d/celery.conf
# Python installs
WORKDIR /proteogyver/resources
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed -r requirements.txt
WORKDIR /proteogyver
RUN sed -i 's\"/home", "kmsaloka", "Documents", "PG_cache"\"/proteogyver", "cache"\g' parameters.json  
RUN sed -i 's\"Local debug": true\"Local debug": false\g' parameters.json  

# Expose ports (jupyterHub. dash)
EXPOSE 8090 8050

# Finished.
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]