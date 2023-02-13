FROM python:3.10.9-slim-bullseye

WORKDIR /usr/src/app

# Enable contrib repo
RUN sed -i'.bak' 's/$/ contrib/' /etc/apt/sources.list

RUN apt-get -y update

# Install fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y fontconfig ttf-mscorefonts-installer
RUN fc-cache -f -v

# Install pdftk
RUN apt-get -y install pdftk

# Install chrome from repo
RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    --no-install-recommends \
    && curl -sSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y \
    google-chrome-stable \
    --no-install-recommends

# Permissions for chrome
RUN groupadd chrome && useradd -g chrome -s /bin/bash -G audio,video chrome \
    && mkdir -p /home/chrome && chown -R chrome:chrome /home/chrome

# Install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "streamlit","run", "./diagramarPrueba.py" ]

