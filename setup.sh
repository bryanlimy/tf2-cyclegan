#!/bin/sh

macOS=false
current_dir="$(pwd)"

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      printf '\nInstalling on macOS\n'
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf '\nInstalling on Linux\n'
      ;;
    *)
      echo '\nOnly Linux and macOS are currently supported.'
      exit 1
      ;;
  esac
}

install_packages() {
  printf '\nInstalling required libraries...\n'
  if [ "$macOS" = "false" ]; then
    conda install -c nvidia cudatoolkit=11.1 cudnn nccl -y
  fi
  pip install -r requirements.txt
}

check_requirements
install_packages

printf '\nSetup completed.'
