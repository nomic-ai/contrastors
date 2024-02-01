SHELL:=/bin/bash -o pipefail
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
APP_NAME:=app
PYTHON:=python3.8


black:
	source env/bin/activate; black src -l 120 -S --target-version py310 .

isort:
	source env/bin/activate; isort  --skip env --profile black --ignore-whitespace --atomic -w 120 .