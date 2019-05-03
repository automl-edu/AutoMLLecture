PYTHONPATH := $(PYTHONPATH):./src:./tests
export PYTHONPATH

all: init test

init:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -p '*test*.py'

.PHONY: init test all
.DEFAULT_GOAL := test
