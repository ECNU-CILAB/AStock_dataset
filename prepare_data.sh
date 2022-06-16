#!/usr/bin/env bash
pip install gdown

mkdir -p datasets
cd datasets

mkdir -p raw
cd raw

gdown "https://drive.google.com/uc?id=1PwWn9dNgczGyFySC5zj0rPp1qA4F04NW"
gdown "https://drive.google.com/uc?id=1XSutgq2N7EhxethiFqnbjMZXhfQw_ZdS"
gdown "https://drive.google.com/uc?id=18xQzXEuxAX4lG8uE_YcYM9_8e6ui7yQ1"
gdown "https://drive.google.com/uc?id=18nA5gpgVUsldJTsP42_uCrIujXJX78h1"
gdown "https://drive.google.com/uc?id=1dfeCd5Thhu6kpfgVX3_1xFFIZPG4S2uu"


