FROM furnari/compute_flow

VOLUME /input
VOLUME /output

ADD compute_flow_wrapper.sh /bin/
ENTRYPOINT ["/bin/compute_flow_wrapper.sh"]
CMD ["frame%06d.jpg"]
