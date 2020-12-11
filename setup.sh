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
  printf '\nInstalling tensorflow...\n'
  if [ "$macOS" = "true" ]; then
    pip install -q tensorflow==2.3.1
  else
    conda install -q cudatoolkit=10.1 cudnn=7.6 cupti=10.1 numpy blas scipy -y
    pip install -q tensorflow==2.3.1
  fi
  printf '\nInstalling other Python packages...\n'
  pip install -q -r requirements.txt
}

check_requirements
install_packages

printf '\nSetup completed.'
