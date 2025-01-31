.PHONY: install
install:
	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: test
test:
	pytest tests -v

.PHONY: build-docker
build-docker:
	docker build -t hdce:latest .

.PHONY: train
train:
	python src/trainer/train.py --config config/default.yaml

.PHONY: inference
inference:
	./scripts/run_inference.sh 