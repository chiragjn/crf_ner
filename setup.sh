#!/bin/bash
pip install -U -r requirements.txt
cd crf_ner/
python download_assets.py
