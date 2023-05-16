FROM python:3.11-slim-bullseye

WORKDIR /usr/src/app

# Enable contrib repo
RUN sed -i'.bak' 's/$/ contrib/' /etc/apt/sources.list

RUN apt-get -y update

# Install fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y fontconfig ttf-mscorefonts-installer --no-install-recommends
RUN fc-cache -f -v

# Install chrome from repo
RUN  apt-get install -y chromium --no-install-recommends

# Install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "streamlit","run", "./diagramarPrueba.py" ]
