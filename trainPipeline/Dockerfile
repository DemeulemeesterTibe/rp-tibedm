ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    wget \
    && apt-get install -y python3-pip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV HOME "/root"
ENV CONDA_DIR "${HOME}/miniconda"
ENV PATH="$CONDA_DIR/bin":$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PIP_DOWNLOAD_CACHE="$HOME/.pip/cache"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh \
    && echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /app

COPY requirements.txt .

RUN conda create --name api python=3.9 \
    && conda activate api \
    && echo "conda activate api" >> "${HOME}/.bashrc" \
    && source "${HOME}/.bashrc" 

RUN pip3 install -r requirements.txt

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 

COPY ./ ./

EXPOSE 9000

# ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

CMD ["python", "trainPipeline.py"]