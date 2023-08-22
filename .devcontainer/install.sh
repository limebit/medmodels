#!/bin/bash

apt update && apt install -y vim
git clone https://github.com/zsh-users/zsh-autosuggestions /home/vscode/.oh-my-zsh/custom/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git /home/vscode/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

if [ "$USE_POWERLEVEL10K" == "true" ]; then
  git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /home/vscode/.oh-my-zsh/custom/themes/powerlevel10k
  sed -i s@robbyrussell@powerlevel10k/powerlevel10k@ /home/vscode/.zshrc
else
  rm /home/vscode/.p10k.zsh
fi

chsh -s /bin/zsh
