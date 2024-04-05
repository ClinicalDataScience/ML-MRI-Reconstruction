FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN mkdir /install

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-utils=2.4.9 curl=7.81.0-1ubuntu1.10 libturbojpeg0-dev=2.1.2-0ubuntu1 \
    libopencv-dev=4.5.4+dfsg-9ubuntu4 bart-cuda=0.7.00-5 \
    graphviz=2.42.2-6 rsync=3.2.7-0ubuntu0.22.04.2 slurm-client=21.08.5-2ubuntu1 tree=2.0.2-1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_common.txt constraints_pytorch.txt requirements_pytorch.txt requirements_pytorch_no_deps.txt /install/

RUN pip install --no-cache-dir pip==23.1.2 && \
    pip install --no-cache-dir -r /install/requirements_pytorch.txt && \
    pip install --no-cache-dir --no-deps -r /install/requirements_pytorch_no_deps.txt

RUN useradd -u 1111 -d /workspace -M -s /bin/bash -p cds cds && \
    chown cds:cds /workspace

USER cds

WORKDIR /workspace

ENV HOME /workspace

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/mpi/lib/

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/workspace --ip 0.0.0.0 --no-browser"]
