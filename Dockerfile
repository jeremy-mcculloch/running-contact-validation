# Pinned FEniCSx/DOLFINx base image (FE-00): DOLFINx 0.11.0.post0, PETSc 3.25.1.
# The digest is the `dolfinx/dolfinx:stable` image verified in GitHub Actions
# run 44432258121 on 2026-11-19. Changing it requires re-running the FE tests
# and updating the version table in README.md.
FROM dolfinx/dolfinx:stable@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
# The repository is bind-mounted at /workspace (see compose.yaml); expose the
# package source without installing it into the image. Prepend, keeping the
# base image's PYTHONPATH (which locates the real-scalar DOLFINx build).
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}
# Inside the container, FE benchmarks must run; missing FEniCSx is an error.
ENV RCV_REQUIRE_FENICSX=1

CMD ["bash"]
