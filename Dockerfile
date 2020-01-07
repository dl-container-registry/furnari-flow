FROM willprice/opencv2-cuda8 
RUN mkdir -p /src /input /output

VOLUME /input
VOLUME /output
# We only have CUDA 8.0 which means we can't build binaries for the newer GPU architectures (above Pascal --
# see https://docs.nvidia.com/cuda/archive/8.0/pascal-compatibility-guide/index.html#application-compatibility-on-pascal)
# This means that the PTX (intermediate code) kernels are compiled at runtime by the CUDA driver. This is a very slow
# process and can introduce around a minute of lag time between the first CUDA kernel invocation and its actual
# execution.
# JIT-ed PTX code is stored in ~/.nv/cache by default, but we can override that with CUDA_CACHE_PATH. To make the JITed
# kernels persistent across container invocations we create a /cache volume to store these files. It is ideal to use a
# quick block device (e.g. SSD) as the backing store for this cache.
VOLUME /cache
ENV CUDA_CACHE_PATH /cache/nv

ADD . /src/compute_flow
WORKDIR /src/compute_flow
RUN mkdir build && \
    cd build && \
    cmake \
        -D OpenCV_DIR=/usr/local/share/OpenCV \
        .. && \
    make -j $(nproc)
RUN cp build/compute_flow /bin

ADD compute_flow_wrapper.sh /bin/
WORKDIR /input
ENTRYPOINT ["/bin/compute_flow_wrapper.sh"]
CMD ["frame_%010d.jpg"]
