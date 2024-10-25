#!/bin/sh

source venv/bin/activate
python tabulate.py
rm -f gilani/* varol/*
find . -name varol*.png -exec mv -f -t varol/ {} +
find . -name gilani*.png -exec mv -f -t gilani/ {} +
