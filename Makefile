SHELL:=bash
CONTAINER_NAME:=willprice/furnari-flow
SINGULARITY_NAME:=furnari-flow.simg
TAG:=

.PHONY: all
all: build singularity

.PHONY: build
build:
	docker build -t $(CONTAINER_NAME) .

.PHONY: push
push:
	docker push $(CONTAINER_NAME)

.PHONY: singularity
singularity: $(SINGULARITY_NAME)

$(SINGULARITY_NAME):
	singularity  build $@ Singularity
