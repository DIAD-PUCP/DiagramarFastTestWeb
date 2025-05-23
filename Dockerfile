FROM python:3.13-slim-bookworm

WORKDIR /usr/src/app

# Enable contrib repo
RUN sed -i'.bak' '/^Components:/s/$/ contrib/' /etc/apt/sources.list.d/debian.sources

RUN apt-get -y update

# Install fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y dumb-init fontconfig ttf-mscorefonts-installer --no-install-recommends
RUN fc-cache -f -v

# Install chrome
RUN apt-get install -y wget
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb

# Install python packages
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

CMD [ "streamlit","run", "./diagramarPrueba.py" ]
