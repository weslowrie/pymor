#!/bin/bash

cd /pymor
~/venv/bin/python setup.py build_ext -i

exec "${@}"
