# For finding latest versions of the base image see
# https://github.com/SwissDataScienceCenter/renkulab-docker
FROM renku/renkulab-py:3.10-0.25.0

# Uncomment and adapt if code is to be included in the image
# COPY src /code/src

# Uncomment and adapt if your R or python packages require extra linux (ubuntu) software
# e.g. the following installs apt-utils and vim; each pkg on its own line, all lines
# except for the last end with backslash '\' to continue the RUN line
#
USER root
RUN apt-get update && \
    apt-get -y upgrade && \                 
    apt-get install -y --no-install-recommends \
    apt-utils \
    vim imagemagick ghostscript

RUN sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml 
USER ${NB_USER}

# For streamlit
COPY jupyter_notebook_config.py ${HOME}/.jupyter/

# install the python dependencies
COPY requirements.txt environment.yml /tmp/
RUN mamba env update -q -f /tmp/environment.yml && \
    /opt/conda/bin/pip install --upgrade setuptools wheel && \
    /opt/conda/bin/pip install -r /tmp/requirements.txt && \
    conda clean -y --all && \
    conda env export -n "root"

# RENKU_VERSION determines the version of the renku CLI
# that will be used in this image. To find the latest version,
# visit https://pypi.org/project/renku/#history.
ARG RENKU_VERSION=2.9.2

# to run streamlit
COPY --chown=jovyan:users jupyter_notebook_config.py ${HOME}/.jupyter/
RUN mkdir /home/jovyan/.streamlit && \
    printf "[general]\nemail = \"\"" > /home/jovyan/.streamlit/credentials.toml

########################################################
# Do not edit this section and do not add anything below

RUN if [ -n "$RENKU_VERSION" ] ; then \
        source .renku/venv/bin/activate ; \
        currentversion=$(renku --version) ; \
        if [ "$RENKU_VERSION" != "$currentversion" ] ; then \
            pip uninstall renku -y ; \
            pip install --force renku==${RENKU_VERSION} ;\
        fi \
    fi

########################################################
