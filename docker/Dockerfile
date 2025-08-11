# syntax=docker/dockerfile:1.7

# Stage 1: build environment
FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build

# Keep the env in a known, relocatable path
ENV PIXI_HOME=/opt/pixi
ENV ENV_PREFIX=${PIXI_HOME}/envs/default

# COPY pixi.lock pyproject.toml /app
COPY . /app/
WORKDIR /app

# Copy only project metadata & lockfiles first

# Install the locked environment (no source yet)
# --locked/--frozen ensures no resolver drift in CI
RUN pixi install --locked -e default
RUN pixi shell-hook -e default > /shell-hook.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh

# # ---- Now add source and extras that must live in the env ----
# COPY src ./src
# COPY README.md LICENSE ./
#
# # # Extra pip/git deps that aren’t in conda-forge:
# # # (install them into the *same* env)
# # RUN python -m pip install --no-deps \
# #     "snaphu>=0.4" \
# #     git+https://github.com/isce-framework/spurt@main
# # # (You said we can ignore whirlwind; keep or remove as needed.)
#
# # Install your package itself (non-editable; smaller)
# # RUN ${ENV_PREFIX}/bin/python -m pip install --no-deps .
#
# # Optional: prune/strip to cut tens to hundreds of MB
# # (strip is present in the pixi image; if not, install binutils in this stage)
# RUN find ${ENV_PREFIX} -type f -name "*.a" -delete && \
#     find ${ENV_PREFIX} -type d -name "tests" -prune -exec rm -rf {} + && \
#     find ${ENV_PREFIX} -type f -name "*.pyc" -delete && \
#     find ${ENV_PREFIX} -type d -name "__pycache__" -prune -exec rm -rf {} + || true
# # Strip shared libs and binaries (best-effort)
# RUN set -e; \
#     if command -v strip >/dev/null 2>&1; then \
#       find ${ENV_PREFIX} -type f \( -name "*.so" -o -perm -111 \) -exec sh -c 'file -b "$1" | grep -q ELF && strip --strip-unneeded "$1" || true' _ {} \; ; \
#     fi

############################
# Stage 2: minimal runtime
############################

FROM ubuntu:24.04 AS production

# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
# COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook.sh /shell-hook.sh
WORKDIR /app
# set the entrypoint to the shell-hook script (activate the environment and run the command)
# no more pixi needed in the prod container
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]

# CMD ["dolphin"]
####
# FROM debian:bookworm-slim AS runtime
#
# # Tiny but safe system deps (certs + basic locale)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends ca-certificates libstdc++6 tini && \
#     rm -rf /var/lib/apt/lists/*
#
# # OCI labels
# LABEL org.opencontainers.image.description="Container for InSAR PS/DS phase analysis using dolphin" \
#       org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>" \
#       org.opencontainers.image.url="https://github.com/isce-framework/dolphin" \
#       org.opencontainers.image.source="https://github.com/isce-framework/dolphin" \
#       org.opencontainers.image.documentation="https://dolphin-insar.readthedocs.io/en/latest/" \
#       org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"
#
# # Copy only the final environment (no caches)
# ENV ENV_PREFIX=/opt/env
# COPY --from=build /opt/pixi/envs/default ${ENV_PREFIX}
#
# # Non-root user
# ARG PIXI_USER_ID=1000
# ARG PIXI_USER_GID=1000
# RUN groupadd -g ${PIXI_USER_GID} pixi && \
#     useradd -m -u ${PIXI_USER_ID} -g ${PIXI_USER_GID} -s /bin/bash pixi && \
#     mkdir -p /work && chown -R pixi:pixi /work
#
# # Wire up the env without activation
# ENV PATH="${ENV_PREFIX}/bin:${PATH}"
# # Some packages expect these:
# ENV CONDA_PREFIX="${ENV_PREFIX}"
# ENV LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}"
# ENV PYTHONNOUSERSITE=1
# WORKDIR /work
# USER pixi
#
# # Use tini as PID 1 for signal handling
# ENTRYPOINT ["/usr/bin/tini","--"]
# CMD ["dolphin","--help"]
#
