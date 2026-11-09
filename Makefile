# Convenience targets for PBP-ReID.

PYTHON ?= python
CONFIG ?= configs/PBP-ReID.yml

.PHONY: help install test smoke compile train test-eval multi efficiency ablation clean

help:
	@echo "make install      - install the Python dependencies"
	@echo "make test         - run the unit tests"
	@echo "make smoke        - run the end-to-end smoke test (no data needed)"
	@echo "make compile      - byte-compile every module"
	@echo "make train        - train on Market-1501 (edit DATASETS.ROOTS first)"
	@echo "make test-eval    - evaluate the last checkpoint"
	@echo "make multi        - train on the seven-dataset union"
	@echo "make efficiency   - parameters, FLOPs, latency and throughput"
	@echo "make ablation     - run the four configurations of Table 5"
	@echo "make clean        - remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

compile:
	$(PYTHON) -m compileall -q config datasets loss model processor solver tools utils
	$(PYTHON) -m compileall -q train.py test.py

test:
	$(PYTHON) -m pytest tests -q

smoke:
	$(PYTHON) tools/smoke_test.py

train:
	$(PYTHON) train.py --config_file $(CONFIG) MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/market1501

test-eval:
	$(PYTHON) test.py --config_file $(CONFIG) MODEL.PRETRAIN_CHOICE none \
		TEST.WEIGHT logs/market1501/pbp_reid_60.pth OUTPUT_DIR logs/market1501

multi:
	$(PYTHON) train.py --config_file configs/PBP-ReID-multiscene.yml \
		MODEL.DEVICE_ID "('0')" OUTPUT_DIR logs/multiscene

efficiency:
	$(PYTHON) tools/count_params_flops.py
	$(PYTHON) tools/measure_speed.py --batch-size 64 --iterations 100

ablation:
	bash bash/run_ablation.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f *.pyc
