FROM mcr.microsoft.com/vscode/devcontainers/python:3.10 as base

ARG USE_POWERLEVEL10K

COPY ./.zshrc /home/vscode/.zshrc
COPY ./.p10k.zsh /home/vscode/.p10k.zsh
COPY ./install.sh ./install.sh

RUN ./install.sh
