FROM pgbase:1.3
LABEL maintainer="Kari Salokas kari.salokas@helsinki.fi"

# First steps
USER root

RUN mkdir -p /proteogyver/data/Server_output
RUN mkdir -p /proteogyver/data/MS_rundata
RUN mkdir -p /etc/supervisor/conf.d
RUN mkdir -p /etc/jupyterhub
RUN mkdir -p /proteogyver/data/unparsed_stats
RUN mkdir -p /proteogyver/data/Server_output/stats

WORKDIR /
# Create mounts for the data
COPY app /proteogyver
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
RUN touch /var/log/cron.log
RUN crontab /etc/cron.d/cron_maintenance_jobs


# Unpack database
RUN cp /proteogyver/resources/celery.conf /etc/supervisor/conf.d/celery.conf

WORKDIR /proteogyver
RUN sed -i 's\"/home", "kmsaloka", "Documents", "PG_cache"\"/proteogyver", "cache"\g' parameters.json  
RUN sed -i 's\"Local debug": true\"Local debug": false\g' parameters.json  

# This will fix a bug in the 0.6 version of dash_uploader. It's a very crude method, but it works for this application.
RUN sed -i 's/isinstance/False:#/g' /usr/local/lib/python3.10/dist-packages/dash_uploader/callbacks.py


# Install miniconda and create conda environment

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
# Python installs
WORKDIR /proteogyver
#RUN pip3 install --upgrade pip
#RUN pip3 install --ignore-installed -r requirements.txt
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#    bash miniconda.sh -b -p /opt/conda && \
#    rm miniconda.sh
#ENV PATH="/opt/conda/bin:${PATH}"

# Create and activate conda environment from yml file
RUN conda env create -f resources/environment.yml
RUN conda clean -afy
#SHELL ["/bin/bash", "-c"]
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate proteogyver" >> ~/.bashrc

COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh
EXPOSE 8090 8050
ENTRYPOINT ["/bin/bash", "/docker_entrypoint.sh"]