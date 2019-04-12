SHELL := /bin/bash
USER_NAME ?= truskovskiyk
USER_PASS ?= $(CI_KYRYL_TOKEN)
KD_IMAGE_NAME ?= kd
IMAGE_TAG ?= latest
DOCKER_REGISTRY ?= registry-staging.neu.ro

.PHONY: setup
setup:
	pip install -r dev-requirements.txt


.PHONY: lint
lint:
	black --check kd tests
	flake8 kd tests
	isort -rc -c kd tests

.PHONY: format
format:
	isort -rc kd tests
	black kd tests

.PHONY: build_and_push_docker_branch
build_and_push_docker_branch:
	docker build -t kd:latest .
	docker login -u $(USER_NAME) -p $(USER_PASS) $(DOCKER_REGISTRY)
	docker tag kd:latest $(DOCKER_REGISTRY)/$(USER_NAME)/$(KD_IMAGE_NAME):$(CIRCLE_SHA1)
	docker push $(DOCKER_REGISTRY)/$(USER_NAME)/$(KD_IMAGE_NAME):$(CIRCLE_SHA1)


.PHONY: build_and_push_docker_master
build_and_push_docker_master:
	docker build -t style:latest .
	docker login -u $(USER_NAME) -p $(USER_PASS) $(DOCKER_REGISTRY)

	docker tag style:latest $(DOCKER_REGISTRY)/$(USER_NAME)/$(KD_IMAGE_NAME):$(IMAGE_TAG)
	docker push $(DOCKER_REGISTRY)/$(USER_NAME)/$(KD_IMAGE_NAME):$(IMAGE_TAG)
