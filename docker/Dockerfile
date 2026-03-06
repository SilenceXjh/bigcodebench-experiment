FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    pkg-config \
    # GDAL / geospatial
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    # Image processing
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Audio processing (librosa / soundfile)
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    ffmpeg \
    # OCR (pytesseract)
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    # Fonts (wordcloud / matplotlib)
    fonts-dejavu-core \
    fonts-liberation \
    # Cryptography
    libssl-dev \
    libffi-dev \
    # Misc
    curl \
    wget \
    git \
    libxml2-dev \
    libxslt1-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# ── Core numeric / scientific ────────────────────────────────────────────────
RUN pip install \
    numpy \
    scipy \
    sympy \
    pandas \
    statsmodels

# ── Machine learning / deep learning ────────────────────────────────────────
RUN pip install \
    scikit-learn \
    scikit-image \
    tensorflow \
    keras

# ── NLP ─────────────────────────────────────────────────────────────────────
RUN pip install \
    nltk \
    gensim \
    textblob \
    wordninja \
    Levenshtein \
    regex

# ── Image / audio / CV ──────────────────────────────────────────────────────
RUN pip install \
    opencv-python-headless \
    Pillow \
    matplotlib \
    seaborn \
    wordcloud \
    librosa \
    soundfile \
    pytesseract

# ── Geospatial ───────────────────────────────────────────────────────────────
RUN pip install \
    geopandas \
    shapely \
    geopy \
    folium

# ── Web / HTTP ───────────────────────────────────────────────────────────────
RUN pip install \
    requests \
    mechanize \
    beautifulsoup4 \
    lxml \
    pyquery \
    wikipedia

# ── Flask ecosystem ──────────────────────────────────────────────────────────
RUN pip install \
    flask \
    flask-login \
    flask-restful \
    flask-wtf \
    flask-mail \
    werkzeug \
    WTForms

# ── Django ───────────────────────────────────────────────────────────────────
RUN pip install django

# ── Data formats / serialization ─────────────────────────────────────────────
RUN pip install \
    openpyxl \
    xlwt \
    python-docx \
    xmltodict \
    PyYAML \
    chardet \
    python-dateutil \
    pytz \
    holidays

# ── Cryptography / security ──────────────────────────────────────────────────
RUN pip install \
    cryptography \
    rsa \
    pycryptodome \
    blake3

# ── Utilities ────────────────────────────────────────────────────────────────
# Note: mpl_toolkits is part of matplotlib, not a standalone pip package
RUN pip install \
    psutil \
    natsort \
    faker \
    prettytable \
    texttable \
    sendgrid \
    python-http-client

# ── Post-install NLTK data ────────────────────────────────────────────────────
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True)"

# ── Post-install TextBlob corpora ────────────────────────────────────────────
RUN python -m textblob.download_corpora

# Verify key imports
RUN python -c "\
import numpy, pandas, scipy, sympy, statsmodels; \
import sklearn, tensorflow, keras; \
import nltk, gensim, textblob; \
import cv2, PIL, matplotlib, seaborn, wordcloud, librosa, soundfile; \
import geopandas, shapely, geopy, folium; \
import requests, bs4, lxml, pyquery; \
import flask, flask_login, flask_restful, flask_wtf, flask_mail, werkzeug; \
import django; \
import openpyxl, docx, xmltodict, yaml, chardet, dateutil, pytz, holidays; \
import cryptography, rsa, Crypto, blake3; \
import psutil, natsort, faker, prettytable, texttable; \
print('All imports OK')"

WORKDIR /app