FROM python:3.11.2-slim-bullseye

WORKDIR /usr/src/app

RUN apt -y update

# Enable contrib repo
RUN sed -i'.bak' 's/$/ contrib/' /etc/apt/sources.list
RUN apt install -y curl gnupg --no-install-recommends
RUN curl -sSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list

RUN apt -y update

# Install fonts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt install -y fontconfig ttf-mscorefonts-installer --no-install-recommends
RUN fc-cache -f -v

# Install pdftk
RUN apt -y install pdftk --no-install-recommends

# Install chrome from repo
RUN  apt install -y google-chrome-stable --no-install-recommends

# Permissions for chrome
RUN groupadd chrome && useradd -g chrome -s /bin/bash -G audio,video chrome \
    && mkdir -p /home/chrome && chown -R chrome:chrome /home/chrome

# Install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "streamlit","run", "./diagramarPrueba.py" ]

