# the docker container for binder needs pymor installed entirely inside
# the container, local dev needs it in path from mounted src
# we trick docker into fulfilling both roles via a conditional ONBUILD
# if you want to use the local dev setup, see docker/docker-compose.yml
ARG PYVER=3.7
ARG BUILD_ENV=binder

FROM pymor/testing:$PYVER as image_binder
ONBUILD ADD . /pymor

FROM pymor/testing:$PYVER as image_dev
ONBUILD RUN echo "dev image uses mounted pymor" && mkdir /pymor
ONBUILD ENV PYTHONPATH=/pymor/src:${PYTHONPATH}

# select "base" image according to build arg
FROM image_${BUILD_ENV}
MAINTAINER rene.fritze@wwu.de

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID
ARG PYMOR_JUPYTER_TOKEN

USER root
RUN useradd -d /home/pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER} && \
    chown -R ${NB_USER} /home/pymor /pymor/
# the devel version of vispy needs npm/nodejs installed
RUN /bin/bash -c "curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
      apt update && apt install -y nodejs && pip install -U virtualenv"
USER ${NB_USER}
RUN /bin/bash -c "virtualenv ~/venv && source ~/venv/bin/activate && \
      git clone --recursive https://github.com/vispy/vispy.git ~/vispy && \
      cd ~/vispy && \
      pip install numpy cython ipywidgets && \
      python setup.py build_py &&      pip install . && \
      jupyter nbextension enable --py widgetsnbextension && \
      jupyter nbextension enable --py vispy "

# at this point setup.py exists only in image_binder
# for the dev image it's only available via the mount at runtime
ADD requirements*.txt /tmp/
ADD .binder/ipython_kernel_config.py /etc/ipython/

RUN /bin/bash -c " source ~/venv/bin/activate && \
    cd /pymor/ && pip install jupyterlab && \
    pip install -r /tmp/requirements-optional.txt && \
    [[ -e /pymor/setup.py ]] && pip install /pymor || echo 'no install needed'"

USER root
ADD .binder/dev-entrypoint.bash /usr/local/bin/
USER ${NB_USER}

ENV JUPYTER_TOKEN=${PYMOR_JUPYTER_TOKEN} \
    USER=${NB_USER} \
    HOME=/home/pymor

ENTRYPOINT []
WORKDIR /pymor/notebooks
