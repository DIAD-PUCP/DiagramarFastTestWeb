FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

# Enable contrib repo
RUN sed -i'.bak' '/^Components:/s/$/ contrib/' /etc/apt/sources.list.d/debian.sources

RUN apt-get -y update

# Install fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y dumb-init fontconfig ttf-mscorefonts-installer --no-install-recommends
RUN fc-cache -f -v

# Install chrome-driver from repo
RUN  apt-get install -y chromium-driver --no-install-recommends

# Install python packages
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

CMD [ "streamlit","run", "./diagramarPrueba.py" ]
