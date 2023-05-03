# This is the image we are going add micromaba to:
ARG BASE=ubuntu:22.04@sha256:7a57c69fe1e9d5b97c5fe649849e79f2cfc3bf11d10bbd5218b4eb61716aebe6
# ARG BASE=debian:latest

FROM $BASE
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Label image following opencontainers image-spec annotations recommendation:
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.description="Container for InSAR PS/DS phase analysis using dolphin"
LABEL org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>"
LABEL org.opencontainers.image.url="https://github.com/opera-adt/dolphin"
LABEL org.opencontainers.image.source="https://github.com/opera-adt/dolphin"
LABEL org.opencontainers.image.documentation="https://dolphin-insar.readthedocs.io/en/latest/"
LABEL org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"


RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Location to install dolphin from the pyproject.toml
WORKDIR /dolphin
# Allow us to create new files in the workdir when pip installing
# RUN chown $MAMBA_USER:$MAMBA_USER /dolphin
# USER $MAMBA_USER

# SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY  docker/specfile.txt /tmp/specfile.txt
RUN conda update -n base conda && \
    conda install --yes --channel conda-forge -n base --file /tmp/specfile.txt && \
    conda clean --all --yes

COPY  . .

# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .

# Run the entrypoint from the /work directory
# This means if they mount a volume to /work, we won't be mixing in the repository code
WORKDIR /work

ENTRYPOINT [ "/usr/bin/tini", "--" ]
# CMD [ "/bin/bash" ]
CMD ["dolphin", "--help"]
