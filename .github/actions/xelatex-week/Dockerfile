# Alpine Docker image version
ARG ALPINE_VERSION="20201218"

FROM alpine:${ALPINE_VERSION}

# Installing packages
# hadolint ignore=DL3018
RUN apk update && \
    apk add --no-cache \
        texlive-full && \
    rm -rf /var/cache/apk/

# No 'xelatex' user
#
# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#user
#
# RUN addgroup -S xelatex && adduser -S -G xelatex xelatex
# USER xelatex

# Not dedicated project directory
#
# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#workdir
#
# WORKDIR /home/xelatex/project

# Entrypoint script
#
# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]