FROM ubuntu:latest

# Installing packages
# hadolint ignore=DL3018
RUN apt update && \
    apt install pdftk -y

# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]