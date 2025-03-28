#!/bin/bash
# Copyright Axelera AI, 2023

# TODO support rootless Docker invocation

# Variables
_self="${0##*/}"

VAR_docker_users=

ARGL_ncpu=$(nproc)
ARGL_ngpu="all"
ARGL_list=false
ARGL_delete=false
ARGL_dry_run=false
ARGL_verbose=false
ARG_tag=

# Transform long options to short ones
_ignore=
for arg in "$@"; do
  shift
  case "$_ignore$arg" in
    "--tag")     set -- "$@" "-t" ;;
    "--cpu")     set -- "$@" "-c" ;;
    "--list")    set -- "$@" "-l" ;;
    "--delete")  set -- "$@" "-d" ;;
    "--verbose") set -- "$@" "-v" ;;
    "--dry-run") set -- "$@" "-n" ;;
    "--help")    set -- "$@" "-h" ;;
    "--")        _ignore="ignore"
                 set -- "$@" "$arg"
                 ;;
    *)           set -- "$@" "$arg"
  esac
done

# Parse command-line options
while getopts ":c:t:lvhdn-" opt; do
  case $opt in
    - )
      break
      ;;
    c )
      ARGL_ncpu="$OPTARG"
      ;;
    d )
      ARGL_delete=true
      ;;
    t )
      ARG_tag="$OPTARG"
      ;;
    l )
      ARGL_list=true
      ;;
    n )
      ARGL_dry_run=true
      ;;
    v )
      ARGL_verbose=true
      ;;
    h )
      echo "Usage:"
      echo "  $_self [options]"
      echo
      echo "Launch Axelera Docker"
      echo
      echo "  -t, --tag TAG      specify Docker tag (default based on YAML)"
      echo "      --cpu [n]      specify number of CPUs (default: 4)"
      echo "  -l, --list         list Axelera Docker images and associated volumes"
      echo "                     (can be combined with --tag)"
      echo "  -d, --delete       delete Axelera Docker images and associated volumes"
      echo "                     (can be combined with --tag, otherwise deletes all)"
      echo "  -v, --verbose      enable verbose output"
      echo "  -n, --dry-run      show, but do not execute the docker run command"
      echo "  -h, --help         display this help and exit"
      echo
      echo "Any arguments following '--' are passed verbatim to 'docker run'"
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      echo "Try '$_self --help' for a list of supported options"
      exit 1
      ;;
    : )
      if [ "$OPTARG" == "c" ]; then
	echo "Missing option argument for --cpu" >&2;
      elif [ "$OPTARG" == "t" ]; then
	echo "Missing option argument for --tag" >&2;
      else
	echo "Missing option argument for -$OPTARG" >&2;
      fi
      exit 1
      ;;
    * )
      echo "Invalid option: -$OPTARG" >&2
      echo "Try '$_self --help' for a list of supported options"
      exit 1
  esac
done

# shellcheck disable=SC2004
shift $((OPTIND-1))

ARGL_extra_args="$*"
$ARGL_verbose && [[ -n "$ARGL_extra_args" ]] && echo -e "Passing extra arguments to 'docker run':\n\t$ARGL_extra_args"

# Source install script to check docker installation is OK and
# obtain configuration settings

VAR_launch_docker=true
VAR_installed_cuda_runtime=
VAR_target_container=
VAR_target_container_tag=
AX_docker_system_component=
STATUS_container=
source "install.sh"

if $ARGL_list; then
  if [[ ! -n "$(which docker)" ]]; then
    echo "Docker not installed"
    exit 1
  fi
  sg "docker" "docker image ls" | grep -E "^(REPOSITORY|axelera/.*${ARG_tag})"
  echo
  exec sg "docker" "docker volume ls" | grep -E "(VOLUME NAME|axelera_${ARG_tag})"
  exit 1
fi

if $ARGL_delete; then
  if [[ ! -n "$(which docker)" ]]; then
    echo "Docker not installed"
    exit 1
  fi
  images=${VAR_target_container}${ARG_tag:+:$ARG_tag}
  images=$(sg "docker" "docker images -q ${images}")
  volumes=$(sg "docker" "docker volume ls" | grep -E "axelera_${ARG_tag}" | awk '{print $2}')
  images=${images//$'\n'/ }
  volumes=${volumes//$'\n'/ }
  echo "Removing Axelera Docker images and associated volumes:"
  echo
  sg "docker" "docker image ls" | grep -E "(^REPOSITORY|${images// /|})"
  echo
  sg "docker" "docker volume ls" | grep -E "(VOLUME NAME|${volumes// /|})"
  if response_is_yes "Confirm delete?"; then
    sg "docker" "docker image rm ${images}"
    sg "docker" "docker volume rm ${volumes}"
  fi
  exit 0
fi

# Check Docker is installed
if needed "$AX_docker_system_component"; then
  echo "Docker installation required"
  echo "First run './install.sh --docker'"
  exit 1
fi

# Check Docker container is available
if ! streq "$STATUS_container" "$STR_ok"; then
  echo "Docker image $VAR_target_container with tag $VAR_target_container_tag not found on system"
  echo "First run './install.sh --docker'"
  exit 1
fi

# Create launch command
launch="docker run --rm --name Axelera --cpus $ARGL_ncpu"
launch="$launch --hostname=docker --add-host=docker:127.0.0.1 --privileged --network=host"

# Map UID/GID and home directory
launch="$launch --user $USER"
launch="$launch -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"
launch="$launch -v /dev:/dev"
launch="$launch -v $HOME:$HOME"

# set up CWD, mapping it to the container if not under $HOME
if [[ "$PWD/" != "$HOME/"* ]]; then
  launch="$launch -v $PWD:$PWD"
fi
launch="$launch -w $PWD"

# map a volume to replace the host .local directory
vol_local="axelera_${VAR_target_container_tag}"
launch="$launch -v ${vol_local}:$HOME/.local"

# X11 display
launch="$launch --ipc host"
launch="$launch --env NO_AT_BRIDGE=1"
launch="$launch --env DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"

# GPU-specific settings
if ! streq "$VAR_installed_cuda_runtime" "$STR_not_detected"; then
  launch="$launch --privileged"
  launch="$launch --gpus $ARGL_ngpu"
fi

# make X11 work for ssh host
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
  XAUTH=/tmp/.docker.xauth_axelera_$VAR_target_container_tag
  if ! $ARGL_dry_run; then
    touch $XAUTH # silence xauth warning
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    chmod 777 $XAUTH
  fi
  launch="$launch -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# Image
launch="$launch -it -a STDOUT -a STDERR $ARGL_extra_args $VAR_target_container:$VAR_target_container_tag"

echo "Launching Docker container"
($ARGL_verbose || $ARGL_dry_run) && echo "$launch"
if $ARGL_dry_run; then
  exit 0
fi
exec sg "docker" "$launch"
