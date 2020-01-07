FROM willprice/opencv2-cuda8 
RUN apt-get update && \
    apt-get install -y git qtbase5-dev

RUN mkdir -p /src /input /output

VOLUME /input
VOLUME /output
ENV CUDA_CACHE_PATH=/cache/nv
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
