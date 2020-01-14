SHELL:=bash
CONTAINER_NAME:=willprice/furnari-flow
SINGULARITY_NAME:=furnari-flow.simg
TAG:=latest
SRC:=src/compute_flow.cpp CMakeLists.txt compute_flow_wrapper.sh

.PHONY: all
all: build singularity

.PHONY: build $(SRC)
build:
	docker build -t $(CONTAINER_NAME):$(TAG) .

.PHONY: push
push:
	docker push $(CONTAINER_NAME):$(TAG)

.PHONY: singularity
singularity: $(SINGULARITY_NAME)

$(SINGULARITY_NAME): $(SRC)
	singularity build $@ Singularity
