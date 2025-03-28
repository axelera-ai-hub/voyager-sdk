#!/bin/bash
# Copyright Axelera AI, 2024

# TODO: check Pipfile generation with ==version
# TODO: grant read permission for the gpg public key file before updating the package index (misconfigured umask)
# TODO: Needed->unaval for missing candidates
# TODO: Docker libs can combine with installer libs
# TODO: install apt packages together rather than one after another
# TODO: system libs better named system packages
# TODO: some installer requirements could be installed by the installer itself
# TODO: STATUS_installed_cuda_runtime may be duplicate of VAR_installed_cuda_runtime
# TODO: some packages should be migrated (no attempt by installer) e.g.
# TODO:   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/migrating-from-1.0.html#migration-1-0
# TODO: dpkg-query can error in broken systems e.g. if ctrl+c on apt update (should catch)
# TODO: When printing activation, advise on how to add needed libs
# TODO: is xargs needed/used consistently

# Disable some shellcheck warnings
# shellcheck disable=SC2034
# shellcheck disable=SC2154
# shellcheck disable=SC2155

# Variables
_self=${0##*/}

SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

STR_unset="Unset"
STR_ok="OK"
STR_system="System"
STR_docker="Docker"
STR_needed="Needed"
STR_reinstall_needed="Reinstall Needed"
STR_upgrade_needed="Upgrade Needed"
STR_docker_needed="Docker"
STR_install="Install"
STR_unsupported="Unsupported"
STR_unavailable="Unavailable"
STR_unused="Unused"
STR_not_detected="Not detected"
STR_not_installed="Not installed"

NUM_pipenv_tasks=4

# Command line arguments
ARG_all=false
ARG_artifacts=false
ARG_media=false
ARG_development=false
ARG_user=
ARG_token=
ARG_no_development=false
ARG_runtime=false
ARG_no_runtime=false
ARG_driver=false
ARG_no_driver=false
ARG_gen_requirements=false
ARG_gen_pipfile=false
ARG_patch_activation=false
ARG_dry_run=false
ARG_docker=false
ARG_docker_system=false
ARG_no_docker_system=false
ARG_gen_dockerfile=false
ARG_print_container=false
ARG_yes=false
ARG_YES=false
ARG_status=false
ARG_activate_env=false
ARG_quiet=false
ARG_allow_as_sudo=false
ARG_verbose=false

STATUS_container=$STR_unset
STATUS_pyenv_root=$STR_unset
STATUS_pyenv_python=$STR_unset
STATUS_penv_pipenv=$STR_unset
STATUS_os_name=$STR_unset
STATUS_base_docker=$STR_unset
STATUS_penv_python=$STR_unset
STATUS_penv_pip=$STR_unset
STATUS_penv_setuptools=$STR_unset
STATUS_penv_wheel=$STR_unset
STATUS_penv_requirements=$STR_unset
STATUS_installed_cuda_runtime=$STR_unset

AX_development_component=$STR_unset
AX_development_where=
AX_development_tasks="0"
AX_development_packages=false
AX_development_repos=
AX_development_help=
read -r -d '' AX_development_help <<EOF
The development environment consists of the Python and system libraries
needed to deploy/compile networks for execution on an AIPU.
EOF

AX_runtime_component=$STR_unset
AX_runtime_where=
AX_runtime_tasks="0"
AX_runtime_packages=false
AX_runtime_repos=
read -r -d '' AX_runtime_help <<EOF
The Axelera runtime libraries are required to build and execute C++
inference applications on the AIPU. These can be installed independently
of the Python development environment if you only need to build and
execute precompiled models and applications.
EOF

AX_driver_component=$STR_unset
AX_driver_where=$STR_system
AX_driver_tasks="0"
AX_driver_packages=false
AX_driver_repos=
read -r -d '' AX_driver_help <<EOF
The Axelera PCIe driver is required to run inference tasks on Axelera
AIPU devices. You must install the driver on host platforms containing
AIPU hardware, even if you plan to use a Docker.
EOF

AX_docker_component=$STR_ok # TODO
AX_docker_where=$STR_docker
AX_docker_tasks="0"
AX_docker_packages=false
AX_docker_repos=

AX_docker_system_component=$STR_ok # TODO
AX_docker_system_where=$STR_system
AX_docker_system_repos=

# File generation components
AX_gen_requirements_component=$STR_unset
AX_gen_requirements_tasks="0"
AX_gen_pipfile_component=$STR_unset
AX_gen_pipfile_tasks="0"
AX_patch_activation_component=$STR_unset
AX_patch_activation_tasks="0"

VAR_deferred_warnings=
VAR_any_errors=false
VAR_docker_version=
VAR_docker_apt=
declare -a VAR_installer_apt_deps=()
declare -a VAR_installer_apt_upgrades=()
VAR_installer_pip_deps=
VAR_installer_pip_upgrades=
VAR_install=false
VAR_groups=""
VAR_any_system_packages=false
VAR_system_repositories=
VAR_docker_repositories=
VAR_requirements=
VAR_diff_requirements=
VAR_pyenv=
VAR_pyenv_root_env=false
VAR_pyenv_in_path=false
VAR_pyenv_set_local=false
VAR_tasks="0"
VAR_tasks_unknown_requirements=false
VAR_need_separator=true
VAR_post_table=false
VAR_i="1"

VAR_installed_cuda_runtime=
VAR_used_cuda_runtime=
VAR_used_container=
VAR_target_container=
VAR_target_container_tag="${ARG_tag}"

VAR_apt_check_system=false
VAR_apt_update_system=false

CMD_return="0"

VENV=${VENV:-"venv"}
ACTIVATE="$VENV/bin/activate"
AX_HOME="${HOME}/.cache/axelera"

SYS_OS_name=$(lsb_release -is 2>/dev/null)
SYS_OS_version=$(lsb_release -rs 2>/dev/null)
SYS_OS_version=${SYS_OS_version//_/-}
SYS_OS_version=${SYS_OS_version//./}
SYS_arch=$(dpkg --print-architecture 2>/dev/null)
SYS_config=${SYS_config:-"cfg/config-${SYS_OS_name,,}-$SYS_OS_version-$SYS_arch.yaml"}

trace() {
  # Print trace of function calls for debugging
  local i pref="TRACE: "
  for (( i=${#FUNCNAME[@]}-1; i; i-- )); do
      echo -n "$pref${FUNCNAME[i]}[${BASH_LINENO[i-1]}]"
      pref=" -> "
  done
  echo ": $1"
}

function urldecode() { : "${*//+/ }"; echo -e "${_//%/\\x}"; }

is_set() {
  if [ -z "$1" ]; then
    false
  else
    true
  fi
}

streq() {
  if [[ "$1" == "$2" ]]; then
    true
  else
    false
  fi
}

in_array() {
  # Is $1 in array $2..$N
  local arr=("${@:2}")
  local found=false
  local var=
  for var in "${arr[@]}"; do
    if streq "$var" "$1"; then
      found=true
    fi
  done
  if $found; then
    true
  else
    false
  fi
}

list() {
  local list=
  if [ -n "$1" ]; then
    list=$(compgen -A variable | grep "^$1")
  fi
  [[ "$1" == "$list" ]] && list=
  echo "$list"
}

is_list() {
  local list=
  list=$(list "$1")
  if is_set "$list"; then
    true
  else
    false
  fi
}

listlen() {
  # Calculate length of list
  local list=$(list "$1")
  local var=
  local num=
  local max="-1"
  for var in $list; do
    num=${var//$1_/}
    num=${num%%_*}
    max=$((num>max?num:max))
  done
  if [[ "$max" -eq "-1" ]]; then
    echo "0"
  else
    echo "$(("$max"+1))"
  fi
}

add_to_list() {
  # Add elements of list $2 to list $1"
  local len=$(listlen "$1")
  local dst=$(list "$1")
  local src=$(list "$2")
  local var=
  for var in $src; do
    eval "$1_$len=\"${!var}\""
    len=$((len+1))
  done
}

list_of_dict() {
  #1: YAML variable associated with a list of dictionaries
  local len=$(listlen "$1")
  local list=
  local var=
  if [[ "$len" -gt "0" ]]; then
    for var in $(seq 0 $((len-1))); do
      list="$list $1_$var"
    done
    echo "${list:1}"
  fi
}

ok() {
  if streq "$1" "$STR_ok"; then
    true
  else
    false
  fi
}

in_container() {
  if [ -f "/.dockerenv" ]; then
    true
  else
    false
  fi
}

is_called_from_docker_launcher() {
  # Docker variable set by script that sources install.sh
  # shellcheck disable=SC2154
  if is_set "$VAR_launch_docker" && $VAR_launch_docker; then
    true
  else
    false
  fi
}

arg_docker() {
  if $ARG_docker || is_called_from_docker_launcher; then
    true
  else
    false
  fi
}

arg_docker_or_file() {
  if arg_docker || $ARG_gen_dockerfile; then
    true
  else
    false
  fi
}

needed() {
  if streq "$1" "$STR_needed" || \
      streq "$1" "$STR_reinstall_needed" || \
      streq "$1" "$STR_upgrade_needed" || \
      streq "$1" "$STR_docker_needed" || \
      streq "$1" "$STR_unavailable"; then
    true
  else
    false
  fi
}

is_dry_arg() {
  if $ARG_dry_run; then
    echo "dry"
  fi
}

where_to_status() {
  if streq "$1" "$STR_system"; then
    echo "$STR_needed"
  elif streq "$1" "$STR_docker"; then
    echo "$STR_docker_needed"
  fi
}

requested_install() {
  local component="AX_$1_component"
  if streq "${!component}" "$STR_install"; then
    true
  else
    false
  fi
}

bold() {
  if [ -t 1 ]; then
    printf "\e[1m%s\e[m" "$1"
  else
    echo -n "$1"
  fi
}

spaces() {
  # print $1 number of spaces
  # shellcheck disable=SC2183
  printf '%*s' "$1" | tr ' ' ' '
}

progress_num() {
  local bar=
  local t=${#VAR_tasks}
  local v=${#VAR_i}
  local pad=$((t-v))
  local n=$VAR_tasks
  # If 10 or more tasks, make 0..9 indent prettier
  if $VAR_tasks_unknown_requirements; then
    n="${VAR_tasks}+"
    printf " "
  elif [[ "$VAR_tasks" -ge "10" ]] && [[ "$VAR_i" -lt "10" ]]; then
    printf " "
  fi
  printf '[%s/%s]' "$VAR_i" "$n"
}

progress() {
  local bar=$(progress_num)
  echo "$bar $*"
}

progress_info() {
  $ARG_dry_run || progress "$@"
}

error_print() {
  bold "ERROR"
  echo ": $*"
} >&2

error() {
  error_print "$@"
  exit 1
}

error_continue() {
  VAR_any_errors=true
  error_print "$@"
}

exit_if_error() {
  if $VAR_any_errors; then
    exit 1
  fi
}

warn() {
  bold "WARNING"
  echo ": $*"
}

warn_defer() {
  VAR_deferred_warnings="$VAR_deferred_warnings"$'\n'"$1"
}

print_deferred_warnings() {
  local warn=
  is_set "$VAR_deferred_warnings" && print_newline
  while IFS= read -r warn; do
    is_set "$warn" && warn "$warn"
  done <<< "$VAR_deferred_warnings"
}

first_word() {
  echo "${1%%[[:space:]]*}"
}

cmd() {
  # Run command and put return code in $1
  # Return true for 0, false otherwise
  local retcode=
  if $ARG_dry_run; then
    echo "$*"
    CMD_return=0
    return
  elif $ARG_verbose; then
    progress "$*"
  fi
  if $ARG_verbose; then
    eval "$*" 2>&1 | grep --line-buffered -v '^\s*$' | sed -e "s|^|$(progress_num) |;";
    retcode=${PIPESTATUS[0]}
  else
    eval "$*" &> /dev/null
    retcode=$?
  fi
  CMD_return=$retcode
  if [[ "$retcode" == 0 ]]; then
    true
  elif ! $ARG_verbose; then
    # Run failed command again to get output
    progress "$*"
    eval "$*" 2>&1 | grep --line-buffered -v '^\s*$' | sed -e "s|^|$(progress_num) |;";
    CMD_return=${PIPESTATUS[0]}
    false
  else
    false
  fi
}

cmd_rm() {
  # Remove file or directory even if owned by another user
  if [ -e "$1" ] && [ ! -O "$1" ]; then
    echo "File or directory '$1' owned by another user (using sudo to remove)"
    if cmd sudo rm -rf "$1"; then
      true
    else
      false
    fi
  elif cmd rm -rf "$1"; then
    true
  else
    false
  fi
}

configure_pyenv() {
  # Ensure pyenv is correctly configured
  # Note that versions > 1.2.26 require additional step
  local pyenv_version=$("$VAR_pyenv" --version 2>/dev/null | sed -e 's/pyenv //')
  if ! $VAR_pyenv_root_env && streq "$1" "dry"; then
    echo "export PYENV_ROOT=\"$HOME/.pyenv\""
  fi
  if ! $VAR_pyenv_in_path; then
    if streq "$1" "dry"; then
      echo "export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
      VAR_pyenv_in_path=true
    elif ! streq "$(which pyenv)" "$VAR_pyenv"; then
      export PATH="$PYENV_ROOT/bin:$PATH"
    fi
  fi
  if ! is_set "$pyenv_version" || dpkg --compare-versions "$pyenv_version" "gt" "1.2.26"; then
    # Only newer pyenvs require this (won't run if not installed)
    if streq "$1" "dry"; then
      echo "eval \"\$(pyenv init --path)\""
    elif [ -f "$VAR_pyenv" ]; then
      eval "$(pyenv init --path)"
    fi
  fi
  if streq "$1" "dry"; then
    echo "eval \"\$(pyenv init -)\""
    echo "eval \"\$(pyenv virtualenv-init -)\""
  elif [ -f "$VAR_pyenv" ]; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
  fi
}

configure_pyenv_python() {
  # Ensure pyenv python is correctly configured
  if ! $VAR_pyenv_set_local; then
    if streq "$1" "dry"; then
      echo "pyenv local $AX_penv_python"
    else
      pyenv local "$AX_penv_python" &> /dev/null || error "Pyenv failed to activate Python $AX_penv_python"
    fi
    VAR_pyenv_set_local=true
  fi
}

apt_install_with_dep_check() {
  local cmd=$1
  local pkg=$(first_word "$2")
  orig_cmd=$cmd
  $ARG_verbose || cmd="$cmd -qqq"
  if [[ "$pkg" != "$2" ]]; then
    # check version specifier
    IFS=" " read -r -a pkg_args <<< "$2"
    if [[ "${#pkg_args[@]}" == 3 ]] && [[ "${pkg_args[1]}" == "==" ]] ; then
      # if exact version is specified, use it
      pkg="${pkg_args[0]}=${pkg_args[2]}"
    else
      echo "Warning: \"$2\" cannot be handled by apt-get, which only handles <pkg>=<version>."
      echo "We will uninstall the current version (since it does not match the request) and install the version from the repository."
    fi
  fi
  if ! cmd "$cmd $pkg"; then
    pkgs=$(python3 -c "import sys; sys.path.insert(0, '.'); import installer_support; installer_support.check_depends(\"$orig_cmd $pkg\")")
    if [ ! -z "$pkgs" ]; then
      if response_is_yes "Installing $pkg failed due to held/conflicting dependencies - attempt to resolve by also updating the dependencies: $pkgs"; then
        cmd "$orig_cmd --allow-change-held-packages $pkgs" || error "Failed to install $pkgs"
      else
        error "Failed to install $pkg"
      fi
    else
      error "Failed to install $pkg"
    fi
  fi
}

check_installer_requirements_met() {
  local ok=true
  # use system pip at this stage as not in virtual env here
  local pip_install="pip install"
  if ! is_set "$VIRTUAL_ENV" && [[ $(id -u) -ne 0 ]]; then
    # If not in a venv, and not root, install required
    # packages with --user flag (no sudo/systemwide effect)
    pip_install="$pip_install --user"
  fi
  declare -a installs
  local apt_update="sudo apt update"
  if is_set "$VAR_installer_apt_deps"; then
    if is_set "$apt_update"; then
      installs+=("${apt_update}")
      apt_update=""
    fi
    for pkg in "${VAR_installer_apt_deps[@]}"; do
      installs+=("sudo DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -y ${pkg}")
    done
    VAR_installer_apt_deps=""
  fi
  if is_set "$VAR_installer_apt_upgrades"; then
    if is_set "$apt_update"; then
      installs+=("${apt_update}")
      apt_update=""
    fi
    for pkg in "${VAR_installer_apt_upgrades[@]}"; do
      installs+=("sudo DEBIAN_FRONTEND=noninteractive apt-get --only-upgrade --no-install-recommends install -y ${pkg}")
    done
    VAR_installer_apt_upgrades=""
  fi
  if is_set "$VAR_installer_pip_deps"; then
    installs+=("$pip_install ${VAR_installer_pip_deps## }")
    VAR_installer_pip_deps=""
  fi
  if is_set "$VAR_installer_pip_upgrades"; then
    installs+=("$pip_install --upgrade ${VAR_installer_pip_upgrades## }")
    VAR_installer_pip_upgrades=""
  fi
  if [ "${#installs[@]}" -ne "0" ]; then
    if response_is_yes "Install/update prerequisite packages required by the installer itself"; then
      for install in "${installs[@]}"; do
        if [[ "$install" =~ "apt-get" ]]; then
          local sep="install -y"
          cmd="${install%${sep} *}${sep}"
          pkgs="${install#*${sep} }"
          apt_install_with_dep_check "$cmd" "$pkgs" || ok=false
        else
          cmd "$install" || ok=false
        fi
      done
    else
      echo "First run the following command(s) to allow the installer to work:"
      for install in "${installs[@]}"; do
        echo "  $install"
      done
      ok=false
    fi
    if ! $ok; then
      error "Installer requirements not met"
    fi
  fi
}


installer_apt_requirement() {
  #1: Package
  #2: ge gt etc (optional)
  #3: version number (optional)
  local ver=
  if ! is_dpkg_installed "$1"; then
    VAR_installer_apt_deps+=($1)
  elif [[ $# -eq 3 ]]; then
    ver=$(dpkg-query --show --showformat '${Version}' "$1")
    if ! dpkg --compare-versions "$ver" "$2" "$3"; then
      echo "$1: Required version $2 $3 (version $ver installed)"
      VAR_installer_apt_upgrades+=($1)
    fi
  fi
}

installer_pip_requirement() {
  # Some packages must be managed with pip
  # e.g. see https://github.com/pypa/pipenv/issues/5088
  #1: Package
  #2: ge gt etc (optional)
  #3: version number (optional)
  local ver=
  if ! is_pippkg_installed "$1"; then
    VAR_installer_pip_deps="$VAR_installer_pip_deps $1"
  elif [[ $# -eq 3 ]]; then
    ver=$(python3 -m pip show pipenv | grep "Version:")
    ver=$(echo "${ver##Version:}" | xargs)
    if ! dpkg --compare-versions "$ver" "$2" "$3"; then
      echo "$1: Required version $2 $3 (version $ver installed)"
      VAR_installer_pip_upgrades="$VAR_installer_apt_upgrades $1"
    fi
  fi
}

print_newline() {
  VAR_post_table=true
  $ARG_quiet || echo
}

get_config_from_yaml() {
  # Read YAML file into environment variables
  local envvars=
  if [[ -f "$1" ]]; then
    if envvars=$(python3 parseyaml.py "$1"); then
      set -a; eval "$envvars"; set +a
    else
      # envvars contains python error message
      # shellcheck disable=2001
      echo "$envvars" | sed -e "s|^|$1: |;";
      exit 1
    fi
  else
    error "$1: File not found"
  fi
}

get_tag_from_yaml() {
  # Get hash of YAML configuration ($1) and requirements ($2)
  # $3 is "docker_hash" or "env_hash"
  local hash=
  local result=
  if [[ -f "$1" ]]; then
    if ! result=$(python3 parseyaml.py "$1" "$3" "$2"); then
      # Returned string contains python error message
      # shellcheck disable=2001
      echo "$result" | sed -e "s|^|$1: |;";
      exit 1
    fi
  else
    error "$1: File not found"
  fi
  echo "$result"
}

installed_cuda_runtime() {
  echo "$STR_not_detected"
  # TODO: decide what to do about CUDA support
  #local cuda=
  #local maj=
  #local min=
  #if is_set "$CUDA_VERSION"; then
  #  if streq "$CUDA_VERSION" "cpu"; then
  #    cuda=
  #  else
  #    cuda=$CUDA_VERSION
  #  fi
  #elif [[ -n "$(which nvidia-smi)" ]]; then
  #  cuda=$(nvidia-smi | grep -Eo "CUDA Version: ([0-9]{1,}\.)*[0-9]{1,}" | sed -e 's/CUDA Version: //')
  #fi
  #if is_set "$cuda"; then
  #  maj=$(echo "$cuda" | cut -f1,1 -d'.')
  #  min=$(echo "$cuda" | cut -f2,2 -d'.')
  #  echo "$maj.$min"
  #else
  #  echo "$STR_not_detected"
  #fi
}

get_arg() {
  local arg="ARG_${1//-/_}"
  echo "$arg"
}

get_no_arg() {
  local arg="ARG_no_${1//-/_}"
  echo "$arg"
}

check_arg_or_no_arg() {
  local arg=$(get_arg "$1")
  local no_arg=$(get_no_arg "$1")
  if ${!arg} && ${!no_arg}; then
    error_continue "You must specify only one of --$1 and --no-$1"
  fi
}

check_only_one_arg() {
  local arg1="ARG_${1//-/_}"
  local arg2="ARG_${2//-/_}"
  if ${!arg1} && ${!arg2}; then
    error "You must specify only one of --$1 and --$2"
  fi
}

check_only_one_of_arg() {
  local vari=
  local varj=
  for vari in "$@"; do
    for varj in "$@"; do
      if ! streq "$vari" "$varj"; then
        check_only_one_arg "$vari" "$varj"
      fi
    done
  done
}

only_one() {
  ! streq "$1" "$2"  && error_continue "You must specify only one of --$1 and --$2"
}

check_no_filegen_arg() {
  $ARG_gen_requirements && only_one "$1" "gen-requirements"
  $ARG_gen_pipfile && only_one "$1" "gen-pipfile"
  $ARG_patch_activation && only_one "$1" "patch-activation"
}

check_no_component_install() {
  $ARG_all && only_one "$1" "all"
  $ARG_development && only_one "$1" "development"
  $ARG_runtime && only_one "$1" "runtime"
  $ARG_driver && only_one "$1" "driver"
}

check_no_docker() {
  arg_docker && only_one "$1" "docker"
}

is_sudo() {
  if [[ ! "$HOME" == "/root" ]] && [[ "$EUID" -eq 0 ]]; then
    true
  else
    false
  fi
}

check_no_sudo() {
  is_sudo && error_continue "Run $_self without 'sudo' to enable option --$1"
}

update_repo_list() {
  #Add repo to list ensuring no duplicates
  #1: repo list
  #2: where
  if streq "$2" "$STR_system"; then
    if [[ ! "$VAR_system_repositories" == *${1}* ]]; then
      VAR_system_repositories="$VAR_system_repositories ${1}"
      inc_tasks VAR_tasks
    fi
  elif streq "$2" "$STR_docker"; then
    VAR_docker_repositories="$VAR_docker_repositories ${1}"
  fi
}

resolve_option() {
  # Resolve whether to install component
  local option="ARG_$1"
  local no_option="ARG_no_$1"
  local component="AX_$1_component"
  local where="AX_$1_where"
  local packages="AX_$1_packages"
  local tasks="AX_$1_tasks"
  local help_var="AX_$1_help"
  local desc=$2
  local is_yes=false
  local repos="AX_$1_repos"
  local help=
  if ! needed "${!component}"; then
    return
  fi
  if streq "${!where}" "$STR_docker"; then
    desc="$desc in Docker"
  fi
  is_set "$help_var" && help=${!help_var}
  if ${!option}; then
    is_yes=true
    eval "$component"=\"$STR_install\"
    VAR_install=true
  elif ! ${!no_option}; then
    if $ARG_yes; then
      $VAR_post_table || print_newline
      is_yes=true
      echo "$desc [Yes]"
    else
      $VAR_post_table || print_newline
      if response_is_yes "$desc" "$help"; then
        is_yes=true
      fi
    fi
  fi
  if $is_yes; then
    eval "$component"=\"$STR_install\"
    VAR_install=true
    inc_tasks VAR_tasks "${!tasks}"
    if ${!packages}; then
      streq "${!where}" "$STR_system" && VAR_any_system_packages=true
    fi
    is_set "${!repos}" && update_repo_list "${!repos}" "${!where}"
  fi
}

set_arg_if_not_unset() {
  local arg="ARG_${1//-/_}"
  local no_arg="ARG_no_${1//-/_}"
  if ! ${!no_arg}; then
    eval "$arg"=\(true\)
  fi
}

print_config_header() {
  if ! $VAR_post_table; then
    printf '+'
    printf '%0.s-' {1..77}
    echo '+'
  else
    echo
  fi
}

print_config_separator() {
  printf '|'
  printf '%0.s=' {1..77}
  echo '|'
}

print_new_component() {
  $VAR_need_separator && print_config_separator
  VAR_need_separator=false
}

print_config_title() {
  printf '| %-76s' "$1"
  echo '|'
}

print_config_dep_list() {
  # 1: Description
  # 2: Lists of dependencies
  local name=$1
  local list=$(list "$2")
  local status=
  local var=
  for var in $list; do
    status=${var//AX/STATUS}
    if needed "${!status}" || $ARG_verbose; then
      print_config_setting "$name" "${!var}" "${!status}"
      name=
    fi
  done
  if ! is_set "$name"; then
    # Have we printed anything
    true
  else
    false
  fi
}

is_dollar_env() {
  if [[ "$1" =~ \\\$ ]]; then
    true
  else
    false
  fi
}

print_component() {
  # 1: AX_component variable
  local desc="$1_description"
  local libs=$(list "${1}_libs")
  local envs=$(list_of_dict "${1}_envs")
  local where="${1}_where"
  local lib=
  local var=
  local env=
  local val=
  local status=
  local any=false
  desc=${!desc}
  # Print component libs
  print_config_dep_list "$desc" "${1}_libs" && any=true
  # Print component repositories
  print_config_needed_repos "$1_repos" "$(where_to_status "${!where}")" && any=true
  # Print environment variables
  $any && desc="Environment"
  if $ARG_verbose || $ARG_status; then
    for var in $envs; do
      env="${var}_name"
      val="${var}_value"
      val="${!val}"
      if is_dollar_env "$val"; then
        # Don't print escape character
        val=${val//\\\$/\$}
      fi
      print_env "$desc" "${!env}" "${val}"
      desc=
      any=true
    done
  fi
  if ! $any; then
    print_config_setting "$desc" "Installed" "$STR_ok"
  fi
}

print_config_line() {
  local lines=
  if ! $VAR_post_table; then
    lines=$(fold "-w76" <<< "$1")
    while IFS= read -r line; do
      printf '| %-75s' "${line:0:75}"
      echo ' |'
    done <<< "$lines"
  else
    echo "$1"
  fi
  VAR_need_separator=true
}

print_config_setting() {
  local name=$1
  local stat=${3::50}
  local len=$((59-${#stat}-2))
  local lines=$(fold "-w$len" <<< "$2")
  local pad=$(spaces 59)
  local line=
  while IFS= read -r line; do
    local desc="$line$pad"
    printf '| %-15s' "${name:0:15}"
    echo "| ${desc::$len} $stat |"
    name=
    stat=$(spaces "${#stat}")
  done <<< "$lines"
  VAR_need_separator=true
}

print_env() {
  local name=$1
  local together="${2} ${3}"
  if [[ "${#together}" -le 58 ]]; then
    print_config_setting "$1" "$2" "$3"
    return
  fi
  local lines=$(fold "-w58" <<< "$together")
  local pad=$(spaces 58)
  local line=
  while IFS= read -r line; do
    local desc="$line$pad"
    printf '| %-15s' "${name:0:15}"
    echo "| ${desc::58} |"
    name=
    stat=$(spaces "${#stat}")
  done <<< "$lines"
  VAR_need_separator=true
}

signcomp() {
  case $1 in
    "==")
      echo "eq"
      ;;
    "!=")
      echo "ne"
      ;;
    ">=")
      echo "ge"
      ;;
    ">")
      echo "gt"
      ;;
    "<=")
      echo "le"
      ;;
    "<")
      echo "lt"
      ;;
  esac
}

is_dpkg_installed() {
  # Is package installed with optional version requirement met
  # 1: package name
  # 2: comparision (optional)
  # 3: version number (optional)
  local ver=
  local name="$1"
  local op="$2"
  local cmp="$3"
  local dpkg_info="$4"
  if [[ "$1" == *.deb ]]; then
    # Local Debian file (query name and version)
    name=$(dpkg --info "$1" 2>/dev/null | grep "Package" | xargs)
    name=$(echo "${name##Package:}" | xargs)
    op="=="
    cmp=$(dpkg --info "$1" 2>/dev/null | grep "Version" | xargs)
    cmp=$(echo "${cmp##Version:}" | xargs)
  fi
  is_set "$dpkg_info" || dpkg_info=`dpkg -l | grep -E "^.i" | awk ' {print $2} '`
  if grep -E "^$name(:|$)" <<< $dpkg_info > /dev/null; then
    if is_set "$op" && is_set "$cmp"; then
      ver=$(dpkg-query --show --showformat '${Version}' "$name")
      if dpkg --compare-versions "$ver" "$(signcomp "$op")" "$cmp"; then
        true
      else
        false
      fi
    else
      true
    fi
  else
    false
  fi
}

is_pippkg_installed() {
  if python3 -m pip show "$1" &> /dev/null; then
    true
  else
    false
  fi
}

print_os() {
  local platform="$SYS_OS_name $SYS_OS_version [$SYS_arch]"
  print_config_setting "O/S" "$platform" "$STATUS_os_name"
}

print_config_requirements() {
  #1 = Name
  #2 = Requirements list
  #3 = Status
  local name="$1"
  local var=
  if is_set "$2"; then
    while IFS="" read -r var || [ -n "$var" ]; do
      print_config_setting "$name" "$var" "$3"
      name=
    done <<< "$2"
  fi
}

print_config_needed_repos() {
  # 1: Varable with list of repositories
  # 2: Status
  local list=${!1}
  local name="Repositories"
  local var_i=
  local var_j=
  local sources=
  for var_i in $list; do
    sources="${var_i}_sources"
    for var_j in $(list "$sources"); do
      print_config_setting "$name" "${!var_j}" "$2"
      name=
    done
  done
  if ! is_set "$name"; then
    # Have we printed anything
    true
  else
    false
  fi
}

get_cuda_str() {
  local cuda=
  local runtime=
  if streq "$VAR_installed_cuda_runtime" "$STR_not_detected"; then
    cuda="$STR_not_detected (using libs for cpu)"
  elif streq "$VAR_installed_cuda_runtime" "$VAR_used_cuda_runtime"; then
    cuda="$VAR_installed_cuda_runtime"
  elif is_set "$VAR_used_cuda_runtime"; then
    if streq "$VAR_used_cuda_runtime" "cpu"; then
      runtime="cpu"
    else
      runtime="runtime $VAR_used_cuda_runtime"
    fi
    cuda="$VAR_installed_cuda_runtime (using libs for $runtime)"
  else
    cuda="$VAR_installed_cuda_runtime"
  fi
  echo "$cuda"
}

get_base_docker_str() {
  local base=
  if is_set "$VAR_used_container"; then
   base="$VAR_used_container"
 else
   base="No base docker container found"
  fi
  echo "$base"
}

print_config() {
  local name="Python libs"
  local name="Axelera AI - Voyager SDK"
  local cuda_docker_deps=
  local repo_status=
  local var=
  print_config_header
  print_config_title "$name"
  print_config_header
  print_os
  print_config_setting "CUDA" "$(get_cuda_str)" "$STATUS_installed_cuda_runtime"
  if ! $ARG_status; then
    if arg_docker_or_file; then
      print_new_component
      if arg_docker; then
        print_config_setting "Docker" "$VAR_docker_version" "$AX_docker_system_component"
        if print_config_dep_list "Dependencies" "AX_docker_system_dependencies"; then
          print_config_needed_repos "AX_docker_system_repos" "$(where_to_status $AX_docker_system_where)"
          print_config_separator
        fi
      fi
      arg_docker && print_config_setting "Container" "$VAR_target_container:$VAR_target_container_tag" "$STATUS_container"
      print_config_setting "Base" "$(get_base_docker_str)" "$STATUS_base_docker"
      print_config_dep_list "Libs" "AX_docker_libs"
      print_config_needed_repos "AX_docker_repos" "$(where_to_status $AX_docker_where)"
    fi
    if ! $ARG_no_development; then
      print_new_component
      if is_sudo; then
        print_config_setting "Development" "Run without 'sudo' for more options" "$AX_development_component"
      else
        print_config_setting "Development" "$PWD" "$AX_development_component"
      fi
    fi
  fi
  if $ARG_gen_requirements || ( ! $ARG_no_development && ! is_sudo ); then
    arg_docker_or_file || $ARG_status || print_config_setting "Pyenv" "$PYENV_ROOT" "$STATUS_pyenv_root"
    if $ARG_gen_requirements || [ ! -f "$AX_penv_requirements" ]; then
      print_config_setting "pipenv" "${AX_penv_pipenv#==}" "$STATUS_penv_pipenv"
    fi
    print_config_setting "Python" "$AX_penv_python" "$STATUS_penv_python"
    if ! $ARG_gen_requirements; then
      print_config_setting "Pip" "${AX_penv_pip#==}" "$STATUS_penv_pip"
      print_config_setting "Setuptools" "${AX_penv_setuptools#==}" "$STATUS_penv_setuptools"
      print_config_setting "Wheel" "${AX_penv_wheel#==}" "$STATUS_penv_wheel"
      if is_set "$VAR_diff_requirements"; then
        if $ARG_status; then
          print_config_setting "Python libs" "Inconsistent, run install.sh" "$STATUS_penv_requirements"
        else
          print_config_requirements "Python libs" "$VAR_diff_requirements" "$STATUS_penv_requirements"
        fi
      else
        print_config_setting "Python libs" "$AX_penv_requirements" "$STATUS_penv_requirements"
      fi
    else
      print_config_setting "Python libs" "$AX_penv_requirements" "$STATUS_penv_requirements"
    fi
  fi
  if ! $ARG_no_development; then
    print_config_dep_list "Dependencies" "AX_dependencies"
    print_config_needed_repos "AX_development_repos" "$(where_to_status "$AX_development_where")"
    arg_docker_or_file || in_container || print_config_dep_list "Pyenv deps" "AX_penv_pyenv_dependencies"
  fi
  if ! $ARG_no_runtime; then
    print_new_component
    print_component "AX_runtime"
  fi
  if ! $ARG_no_driver; then
    print_new_component
    print_component "AX_driver"
  fi
  print_config_header
}

print_next_steps() {
  local list=$(list "AX_next")
  local any=false
  local var=
  $VAR_post_table && print_newline
  if arg_docker; then
    var="$1. To launch the container, type:"
    print_config_line "$var"
    print_config_line
    print_config_line "  ./launch-docker.sh"
    print_config_line
  else
    var="$1. To activate the environment, type:"
    print_config_line "$var"
    print_config_line
    print_config_line "  source venv/bin/activate"
    print_config_line
  fi
  for var in $list; do
    if streq "${!var}" "None"; then
      print_config_line
    else
      print_config_line "${!var}"
    fi
  done
  print_config_header
}

print_envs_to_source() {
  local list=$(list_of_dict "$1")
  local var=
  local env=
  local val=
  for var in $list; do
    env="${var}_name"
    env="${!env}"
    val="${var}_value"
    val="${!val}"
    if arg_docker_or_file; then
      if ! is_dollar_env "${val}"; then
        write_to_dockerfile "ENV ${env} \"${val}\""
        eval local ${env}=\"${val}\"
      fi
    else
      echo "${env}=\"${val}\""
    fi
  done

  for var in $list; do
    env="${var}_name"
    env="${!env}"
    val="${var}_value"
    val="${!val}"
    if arg_docker_or_file; then
      if is_dollar_env "${val}"; then
        eval val=\"${val//\\\$/\$}\"
        eval local ${env}=\"${val}\"
        write_to_dockerfile "ENV ${env} \"${val}\""
      fi
    fi
  done
}

reload_python_requirements() {
  VAR_requirements=$(awk NF 2>/dev/null < "$AX_penv_requirements")
}

load_python_requirements() {
  if ! streq "$VAR_used_cuda_runtime" "cpu"; then
    AX_penv_requirements="${AX_penv_requirements%".txt"}-cuda-$VAR_used_cuda_runtime.txt"
  fi
  reload_python_requirements
}

write_to_dockerfile() {
  if ! $ARG_dry_run; then
    echo "$*" >> Dockerfile
  fi
}

set_status_os() {
  if streq "$AX_os_name" "$SYS_OS_name"; then
    STATUS_os_name=$STR_ok
    if streq "$AX_os_version" "$OS_version"; then
      STATUS_os_version=$STR_ok
    else
      STATUS_os_version=$STR_ok
    fi
  else
    STATUS_os_name=$STR_unsupported
  fi
}

check_docker_migration() {
  local list=$(list "AX_docker_migrate")
  local var=
  local any_old=false
  for var in $list; do
    if is_dpkg_installed "${!var}"; then
      warn_defer "${!var}: Package for previous verison of Docker found on system"
      any_old=true
    fi
  done
  if $any_old; then
    warn_defer "You are strongly recommended to first remove all old components"
    warn_defer "before running this installer"
  fi
}

set_status_docker() {
  # Status of docker (via which) may be different
  # from status of component (plus libs)
  if [[ -n "$(which docker)" ]]; then
    VAR_docker_version=$(docker --version)
    AX_docker_component=$STR_ok
  else
    VAR_docker_version=$STR_not_installed
    AX_docker_component=$STR_needed
  fi
  set_status_system_libs "AX_docker_libs" AX_docker_component
  set_status_system_libs "AX_docker_system_dependencies" AX_docker_system_component
  if ! streq "$AX_docker_system_component" "$STR_ok"; then
    # If docker system component is not installed
    # check for conflicts from previous versions
    check_docker_migration "AX_docker_migrate" "AX_docker_system_component"
  fi
}

set_status_pyenv() {
  # 1: COMPONENT variable with combined status
  local list=$(list_of_dict "AX_penv_pyenv")
  STATUS_pyenv_root=$STR_ok
  local status=
  local var=
  local url=
  local dir=
  for var in $list; do
    dir="${var}_dir"
    status=${var//AX/STATUS}
    if [ -d "$PYENV_ROOT/${!dir}" ]; then
      eval "$status"=\"$STR_ok\"
    else
      eval "$status"=\"$STR_needed\"
      inc_tasks "${1//component/tasks}"
      STATUS_pyenv_root=$STR_needed
    fi
  done
  if streq "$STATUS_pyenv_root" "$STR_needed"; then
    eval "$1"=\"$STR_needed\"
  fi
}

selected_cuda_runtime() {
  # Determine compatible cuda libs are available for runtime $1
  # 1: installed CUDA
  # 2: attribute to return
  local installed="$1"
  local attribute="$2"
  local cuda=$(list_of_dict "AX_penv_cuda")
  local cuda_versions=
  local found=false
  local var=
  for var in $cuda; do
    cuda_versions="${var}_version"
    IFS=", " read -r -a cuda_versions <<< "${!cuda_versions}"
    if in_array "$installed" "${cuda_versions[@]}"; then
      found=true
      attribute="${var}_$attribute"
      echo "${!attribute}"
      break
    fi
  done
}

revise_pipfile_for_compat_cuda() {
  # Update requirements for specified CUDA version
  local lib="$1"
  local compat="$2"
  local cuda=$(list_of_dict "AX_penv_cuda")
  local cuda_compat=
  local replace=
  local var=
  local found=false
  for var in $cuda; do
    cuda_compat="${var}_compat"
    cuda_compat=${!cuda_compat}
    if streq "$compat" "$cuda_compat"; then
      replace="${var}_${lib}"
      replace=${!replace}
      is_set "$replace" && found=true
      break
    fi
  done
  # Replace CPU lib with CUDA-specific lib
  if $found; then
    sed -i "s|^${lib} .*$|${lib} = ${replace}|g" Pipfile
  else
    error "$SYS_config: CUDA $cuda_compat not found (compat attribute for '$lib')"
  fi
}

set_status_pip() {
  # Check if already-installed pip meets required version
  local venv_pip=$($VENV/bin/pip3 --version 2>/dev/null)
  venv_pip=$(echo "$venv_pip" | awk '{print $2}')
  if [ -z "$AX_penv_pip" ]; then
    AX_penv_pip=$(latest_available pip $VENV/bin/python3)
    AX_penv_pip=${AX_penv_pip:-$venv_pip}
  fi
  AX_penv_pip=$(sanitize_version "$AX_penv_pip")
  if ! streq $(sanitize_version "$venv_pip") "$AX_penv_pip" || [ -z "$AX_penv_pip" ]; then
    STATUS_penv_pip=$STR_needed
    eval "$1"=\"$STR_needed\"
  fi
}

current_pip_package_version() {
  # Get current version of pip package
  # 1: package name
  # 2: which python to check
  local version=$($2 -m pip show $1 2>/dev/null | grep "Version")
  echo ${version##Version:}
}

set_status_pip_package() {
  # Check if package is installed for the specific python installation
  # 1: package name
  # 2: COMPONENT variable with combined states
  # 3: which python to check
  local current_version=$(current_pip_package_version $1 $3)
  local AX_penv_version_name="AX_penv_$1"
  AX_penv_version=${!AX_penv_version_name}
  if [ -z "$AX_penv_version" ]; then
    AX_penv_version=$(latest_available $1 $3)
    AX_penv_version=${AX_penv_version:-$current_version}
  fi
  AX_penv_version=$(sanitize_version "$AX_penv_version")
  eval "$AX_penv_version_name"=\"$AX_penv_version\"
  if ! streq $(sanitize_version "$current_version") "$AX_penv_version" || [ -z "$AX_penv_version" ]; then
    eval "STATUS_penv_$1"=$STR_needed
    eval "$2"=\"$STR_needed\"
    inc_tasks "${2//component/tasks}"
  fi
}

set_status_pipenv() {
  STATUS_penv_pipenv=$STR_ok
  set_status_pip_package pipenv $1 "$PYENV_ROOT/versions/$AX_penv_python/bin/python3"
}

set_status_gen_requirements() {
  # Determine status of components needed to generate requirements
  # 1: COMPONENT variable with combined status
  AX_gen_requirements_component=$STR_needed
  STATUS_penv_requirements=$STR_needed
  set_status_pyenv AX_gen_requirements_component
  set_status_pipenv AX_gen_requirements_component
  if [ -f "$PYENV_ROOT/versions/$AX_penv_python/bin/python3" ]; then
    STATUS_pyenv_python=$STR_ok
    STATUS_penv_python=$STR_ok
  else
    STATUS_pyenv_python=$STR_needed
    STATUS_penv_python=$STR_needed
    inc_tasks AX_gen_requirements_tasks
  fi
  inc_tasks AX_gen_requirements_tasks "$NUM_pipenv_tasks"
}

latest_available() {
  # Get latest available version of package
  # 1: package name
  # 2: optional python executable, else python3
  local version=$(${2-python3} -m pip install --upgrade --dry-run "$1" 2>/dev/null | grep "Would install" | awk "{ print \$3 }")
  echo "${version#$1-}"
}

sanitize_version() {
  # Sanitize version number for pip
  if [[ "$1" =~ ^[0-9]+ ]]; then
    # Direct version number
    echo "==$1"
  elif [[ "$1" =~ ^\{.*[[:space:]]*version[[:space:]]*=[[:space:]]*\"(.*)\".*\}$ ]]; then
    # extract version from Pipfile format { version = "..." }
    echo ${BASH_REMATCH[1]}
  else
    # pass through assuming already suitable for pip
    echo "$1"
  fi
}

set_status_development() {
  # Determine status of development environment and subcomponents
  local venv_python=
  local venv_req=
  local var_normalized_requirements=
  local var=
  local match=
  local save=
  local diff=
  AX_development_component=$STR_ok
  STATUS_penv_python=$STR_ok
  STATUS_penv_pip=$STR_ok
  STATUS_penv_setuptools=$STR_ok
  STATUS_penv_wheel=$STR_ok
  STATUS_penv_requirements=$STR_ok
  if arg_docker_or_file; then
    if needed "$STATUS_container"; then
      STATUS_penv_python=$STR_docker_needed
      STATUS_penv_pip=$STR_docker_needed
      STATUS_penv_requirements=$STR_docker_needed
      AX_development_component=$STR_docker_needed
    fi
  elif ! is_sudo; then
    # Check if pyenv is installed
    set_status_pyenv AX_development_component
    # Check if pyenv python version required by venv exists
    venv_python=$($VENV/bin/python3 --version 2>/dev/null)
    venv_python="${venv_python#Python }"
    if [ -f "$PYENV_ROOT/versions/$AX_penv_python/bin/python3" ]; then
      STATUS_pyenv_python=$STR_ok
    else
      STATUS_pyenv_python=$STR_needed
      AX_development_component=$STR_needed
    fi
    # Check if we have existing venv with correct Python
    # version installed or available
    if ! streq "$AX_penv_python" "$venv_python"; then
      STATUS_penv_python=$STR_needed
      AX_development_component=$STR_needed
    fi
    # Check if pip/setuptools/wheel are consistent with requirements
    set_status_pip AX_development_component
    set_status_pip_package setuptools AX_development_component "$VENV/bin/python3"
    set_status_pip_package wheel AX_development_component "$VENV/bin/python3"
    if needed "$STATUS_penv_setuptools" || needed "$STATUS_penv_wheel"; then
      STATUS_penv_pip=$STR_needed
      AX_development_component=$STR_needed
    fi
    # Check if already-installed libs are consistent with requirements
    venv_req=$("$VENV/bin/pip" freeze 2>/dev/null)
    if ! is_set "$venv_req"; then
      STATUS_penv_requirements=$STR_needed
      AX_development_component=$STR_needed
    else
      # PEP426 requires comparisions of distribution names to be case
      # insensitive and consider hyphens and underscores equivalent
      save=$VAR_requirements
      var_normalized_requirements=${VAR_requirements,,}
      var_normalized_requirements=${var_normalized_requirements//_/-}
      venv_req=${venv_req,,}
      venv_req=${venv_req//_/-}
      # As a special case, pillow-simd is intended to be
      # interchangeable with pillow
      if [[ "$venv_req" =~ (pillow-simd==)(.*)(.post[^$'\n']) ]]; then
        match=${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}
        var="pillow==${BASH_REMATCH[2]}"
        venv_req=${venv_req//$match/$var}
      fi
      # If a local wheel is followed by comments (with name=version),
      # replace path with comment for accurate diffing with the
      # installed version displayed by pip freeze
      VAR_requirements=
      while IFS="" read -r var || [ -n "$var" ]; do
        if [[ "$var" =~ ^.*.whl[[:space:]]*\#.*$ ]]; then
          var=$(echo "${var##*\#}" | xargs)
        fi
        VAR_requirements="$VAR_requirements"$'\n'"${var}"
      done <<< "$var_normalized_requirements"
      var_normalized_requirements=${VAR_requirements:1}
      VAR_requirements=$save
      diff=$(diff --old-line-format='-%L' --new-line-format="+%L" --unchanged-line-format="" <(echo "$var_normalized_requirements" | sort) <(echo "$venv_req" | sort))
      while IFS="" read -r var || [ -n "$var" ]; do
        # Allow additional files (+) but not removed files (-)
        # requirements specified with @ cannot be diffed easily
        # (pip freeze cannot produce '@ git hashes')
        # so don't consider these also (which can mask issues)
        if [[ "${var::1}" == "-" ]] && ! [[ "${var:1}" =~ ^.*@.*(http(://|s://)|git+ssh://)*$ ]]; then
          VAR_diff_requirements="$VAR_diff_requirements"$'\n'"${var:1}"
          STATUS_penv_requirements=$STR_needed
          AX_development_component=$STR_needed
        fi
      done <<< "$diff"
      VAR_diff_requirements=${VAR_diff_requirements:1}
    fi
    # If a new Python, pip or requirements are required
    # we must install full venv
    if needed $STATUS_penv_python; then
      ok "$STATUS_penv_pip" && STATUS_penv_pip=$STR_reinstall_needed
      ok "$STATUS_penv_requirements" && STATUS_penv_requirements=$STR_reinstall_needed
    fi
    if needed "$STATUS_penv_pip"; then
      ok "$STATUS_penv_python" && STATUS_penv_python=$STR_reinstall_needed
      ok "$STATUS_penv_requirements" && STATUS_penv_requirements=$STR_reinstall_needed
    fi
    if needed "$STATUS_penv_requirements"; then
      ok "$STATUS_penv_python" && STATUS_penv_python=$STR_reinstall_needed
      ok "$STATUS_penv_pip" && STATUS_penv_pip=$STR_reinstall_needed
    fi
    # Determine number of install tasks
    needed "$AX_development_component" && inc_tasks AX_development_tasks # patch venv script
    needed "$STATUS_pyenv_python" && inc_tasks AX_development_tasks
    needed "$STATUS_penv_python" && inc_tasks AX_development_tasks
    needed "$STATUS_penv_pip" && inc_tasks AX_development_tasks
    if needed "$STATUS_penv_requirements"; then
      if ! is_set "$VAR_requirements"; then
        # Need to run pipenv to calculate requirements
        inc_tasks AX_development_tasks "$NUM_pipenv_tasks"
        VAR_tasks_unknown_requirements=true
      fi
    fi
    # If there is no requirements file, check if pipenv
    # is available to generate
    [ -f "$AX_penv_requirements" ] || set_status_pipenv AX_development_component
    set_status_system_libs "AX_penv_pyenv_dependencies" AX_development_component
  fi
  set_status_system_libs "AX_dependencies" AX_development_component
}

is_package_marked_for_removal() {
  if dpkg -l | grep -E "^(r|p)i" | awk ' {print $2} ' | grep -E "^$1(:|$)" > /dev/null; then
    true
  else
    false
  fi
}

add_apt_repositories() {
  #1: List of repositories
  #2: where
  local list=${!1}
  local apt_list=
  local url=
  local key=
  local sources=
  local m1=
  local m2=
  local c1=
  local m2=
  local e1=
  local var_i=
  local var_j=
  local str=
  local save_sym=
  local overwrite_gpg=$([ $ARG_yes ] && echo "--yes" || echo "" )
  for var_i in $list; do
    apt_list="${var_i}_list"
    apt_list=${!apt_list}
    url="${var_i}_gpg_url"
    url=${!url}
    key="${var_i}_gpg_key"
    key=${!key}
    sources="${var_i}_sources"
    # TODO there may be issues if specified in locations where parent directory is not readable
    m1="mkdir -p \"$(dirname "$key")\""
    m2="chmod -R 0755 \"$(dirname "$key")\""
    c1="sh -c 'curl -fsSL \"${url}\" | gpg --dearmor ${overwrite_gpg} -o \"${key}\"'"
    c2="chmod a+r \"$key\""
    m3="mkdir -p \"$(dirname "$apt_list")\""
    e1="deb [arch=$SYS_arch signed-by=\"${key}\"]"
    if streq "$2" "$STR_docker"; then
      write_to_dockerfile "RUN apt update && apt-get install -y --no-install-recommends -y curl ca-certificates gnupg cmake &&\\"
      write_to_dockerfile " $m1 &&\\"
      write_to_dockerfile " $m2 &&\\"
      write_to_dockerfile " $c1 &&\\"
      write_to_dockerfile " $c2 &&\\"
      write_to_dockerfile " $m3 &&\\"
      save_sym=">"
      for var_j in $(list "$sources"); do
        str="$str &&\\"$'\n'" sh -c 'echo \"$e1 ${!var_j}\" $save_sym \"${apt_list}\"'"
        save_sym=">>"
      done
      str=${str#*$'\n'}
      write_to_dockerfile "$str"
    else
      progress_info "Add APT repository $apt_list"
      cmd sudo "$m1" || error "Failed to mkdir $(dirname "$key")"
      cmd sudo "$m2" || error "Failed to mkdir $(dirname "$key")"
      cmd_rm -f "$key" || error "Failed to remove $key"
      cmd sudo "$c1" || error "Failed to install key at ${url} to ${key}"
      cmd sudo "$c2" || error "Failed to set a+r permission for ${key}"
      cmd sudo "$m3" || error "Failed to mkdir $(dirname "$apt_list")"
      save_sym=">"
      for var_j in $(list "$sources"); do
        var_j="${var_j/\$/"\\$"}"
        cmd sudo "sh -c 'echo \"$e1 ${!var_j}\" $save_sym \"${apt_list}\"'" || error "Failed to update $apt_list"
        save_sym=">>"
      done
      complete_task
    fi
  done
}

apt_check_system() {
  if ! $VAR_apt_check_system; then
    progress_info "Check system for broken dependencies"
    if ! cmd sudo apt-get check; then
      print_newline
      error "Please fix reported issue and then re-run this installer"
    fi
    complete_task
    VAR_apt_check_system=true
  fi
}

apt_update_system() {
  local cmd="apt update"
  if ! $VAR_apt_update_system; then
    progress_info "Make sure apt is up-to-date"
    $ARG_verbose || cmd="$cmd -qqq"
    if ! cmd "sudo $cmd"; then
      print_newline
      echo "APT update failed with return code $CMD_return"
      echo "Please resolve manually and then re-run the installer"
      exit 1
    fi
    complete_task
    VAR_apt_update_system=true
  fi
}

install_system_package_with_apt() {
  # 1: package name
  local cmd=
  local pkg=$(first_word "$1")
  apt_check_system
  apt_update_system
  if is_package_marked_for_removal "$pkg"; then
    progress_info "Mark desired state of '$description' package to 'install'"
    sudo "echo \"$pkg\" install | sudo dpkg --set-selections" || error "Failed to update package '$pkg' state to 'install'"
  else
    progress_info "Install $pkg"
    if [[ "$pkg" =~ "axelera-pcie-driver" ]] || [[ "$pkg" =~ "metis-dkms" ]]; then
      cmd='dpkg -l | grep "axelera-pcie-driver" | awk "{ print \$2 }" | xargs -r sudo dpkg -P'
      cmd "$cmd" || error "Failed to remove previous axelera-pcie-driver"
      cmd='dpkg -l | grep "metis-dkms" | awk "{ print \$2 }" | xargs -r sudo dpkg -P'
      cmd "$cmd" || error "Failed to remove previous metis-dkms"
      cmd "sudo modprobe -r metis &> /dev/null" || true
      cmd "sudo rm -f /lib/modules/`uname -r`/{extra,updates}/metis.ko"
    fi
    if [[ "$pkg" =~ \.deb$ ]]; then
      cmd "sudo dpkg -i $pkg"
    else
      cmd="sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
      apt_install_with_dep_check "$cmd" "$1" # pass $1 instead of $pkg in case it includes a version specifier
    fi
    if [[ "$pkg" =~ "axelera-pcie-driver" ]] || [[ "$pkg" =~ "metis-dkms" ]]; then
      if ! in_container; then
        cmd='(sudo /lib/systemd/systemd-udevd --daemon && sudo udevadm control --reload && sudo udevadm trigger && sudo udevadm settle)'
        cmd "$cmd" || echo "Warning: failed to reload udev rules"
      fi
    fi
  fi
  complete_task
}

install_docker_packages() {
  local name=$1
  local list=$(list "$1")
  local status=
  for var in $list; do
    status=${var//AX/STATUS}
    needed "${!status}" && VAR_docker_apt="$VAR_docker_apt ${!var}"
  done
}

install_system_packages_with_apt() {
  local name=$1
  local list=$(list "$1")
  local status=
  for var in $list; do
    status=${var//AX/STATUS}
    needed "${!status}" && install_system_package_with_apt "${!var}"
  done
}

git_checkout() {
  if [ -d "$2" ]; then
    [ -d "$2/.git" ] || warn "Directory '$2' exists but directory '$2/.git' does not exist"
  else
    progress_info "Checkout ${1##*/}"
    cmd git clone --quiet "$1" "$2"
  fi
}

install_pyenv_to_system() {
  local list=$(list_of_dict "AX_penv_pyenv")
  local status=
  local var=
  local url=
  local dir=
  for var in $list; do
    url="${var}_url"
    dir="${var}_dir"
    status=${var//AX/STATUS}
    if needed "${!status}"; then
      git_checkout "${!url}" "$PYENV_ROOT/${!dir}" || error "Failed to checkout ${!url}"
      complete_task
    fi
  done
  install_system_packages_with_apt "AX_penv_pyenv_dependencies"
}

install_python_to_pyenv() {
  local args="pyenv install $AX_penv_python"
  local verbose=
  $ARG_verbose && verbose="--verbose"
  if needed "$STATUS_pyenv_python"; then
    progress_info "$args"
    cmd "$args" "$verbose" || error "Failed to install Python $AX_penv_python with pyenv"
    complete_task
  fi
}

install_pipenv_to_system() {
  local args="python3 -m pip install --user"
  if needed "$STATUS_penv_pipenv"; then
    if streq "$STATUS_penv_pipenv" "$STR_upgrade_needed"; then
      args="$args --upgrade"
    fi
    args="$args \"pipenv$AX_penv_pipenv\""
    progress_info "$args"
    cmd "$args" ||  error "Failed to install pipenv"
    complete_task
  fi
}

install_python_prerequisites() {
  # Install python libraries from prerequisites file with supported formats:
  # lib==version
  # lib @ url
  local lib=
  progress_info "Preinstalling python prerequisites"
  while IFS= read -r lib; do
    lib=${lib%%\#*}
    if [[ "$lib" =~ ^.*==.*$ ]] || [[ "$lib" =~ ^.*@.*(http(://|s://)|git+ssh://)*$ ]]; then
      progress_info "Preinstall $lib"
      if ! cmd python3 -m pip --disable-pip-version-check install --no-deps "\"$lib\""; then
        cmd python3 -m pip debug --verbose # print compatible wheel tags
        echo
        warn "Failed to install $lib"
        echo
        VAR_any_errors=true
      fi
    fi
    complete_task
  done < "$AX_penv_trimmed_requirements"
}

install_pip_extra(){
  # install unconditionally, but if we also knew we needed it, decrement task counter
  local status="STATUS_$1"
  if needed "${!status}"; then
    complete_task
  fi
  local version="AX_penv_${1}"
  progress_info "Install $1 ${!version#==}"
  version="${1}${!version}"
  cmd python3 -m pip install --disable-pip-version-check --upgrade "\"${version}\"" || error "Failed to install \"${version}\""
}

get_env_tag() {
  local env_tag=$(get_tag_from_yaml "$SYS_config" "$AX_penv_requirements" "env_hash")
  echo "${env_tag}"
}

construct_auth_url() {
  # Construct by embedding the authentication details
  # 1: url

  if [ -z "$ARG_user" ]; then
    read -p "
    Email address not provided through --user <user>.
    Please refer to docs/tutorials/install.md for more information.
    Email address:" ARG_user
  fi
  if [ -z "$ARG_token" ]; then
    read -p "
    Token not provided through --token <token>.
    Please refer to docs/tutorials/install.md for more information.
    Token:" -s ARG_token
  fi

  echo $1 | sed -E "s#(https?://)(.*)#\1${ARG_user}:${ARG_token}@\2#"
}

install_python_libs() {
  # Install the libraries from a given category
  # 1: category
  local prefix="AX_penv_${1}_libs"
  local list=$(list "${prefix}")
  local index_url_var="AX_penv_${1}_index_url"
  local index_url=${!index_url_var}
  local auth="AX_penv_${1}_requires_auth"

  shopt -s extglob
  list="${list[@]//${prefix}_+([0-9])_/}"
  shopt -u extglob

  [ "${!auth}" = "True" ] && index_url=$(construct_auth_url $index_url)

  cmd_stem="python3 -m pip --disable-pip-version-check install --index-url ${index_url} --no-deps -c ${AX_penv_requirements}"

  inc_tasks VAR_tasks "$(wc -l <<< "${list}"  | awk NF)"
  progress_info Installing ${1} libraries
  for lib in ${list[@]}; do
    progress_info "Install $lib"
    if ! cmd ${cmd_stem} "\"$lib\""; then
      echo
      warn "Failed to install $lib"
      echo
      VAR_any_errors=true
    fi
    complete_task
  done
}

_to_be_stripped=()

drop_package() {
  local new_list=()
  dropped=false
  local match=
  local pkg="$1"
  for match in "${to_be_stripped[@]}"; do
    if [[ "$pkg" =~ ^$match[=\ ] ]]; then
      dropped=true
    else
      new_list+=("$match")
    fi
  done
  to_be_stripped=("${new_list[@]}")
  $dropped
}

revise_requirements() {
  # tidy up requirements file to be a constraints file
  # and generate a trimmed version without AX packages
  #
  # if $1 is set, increment task counter
  local do_tasks=${1:-false}
  rm -rf "$AX_penv_trimmed_requirements"
  local revised=
  local new_line=
  local pkg=

  list="$(list AX_penv_development)"$'\n'"$(list AX_penv_runtime)"$'\n'"$(list AX_penv_axeleracommon)"

  shopt -s extglob
  for pkg in $list; do
    pkg="${pkg//AX_penv_*+([0-9])_/}"
    pkg="${pkg//_/[_-]}"
    to_be_stripped+=("$pkg")
  done
  shopt -u extglob

  while IFS= read -r pkg; do
    if [[ "$pkg" =~ ^([^#]+)( @ .*)$ ]]; then
      new_line="${BASH_REMATCH[1]} #${BASH_REMATCH[2]}"
    else
      new_line="$pkg"
    fi
    revised="${revised}"$'\n'"${new_line}"
    if [[ "$pkg" =~ ^(.*)\ \#( @ .*)$ ]]; then
      # uncomment url packages if already commented
      pkg="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    fi
    if ! drop_package "$pkg"; then
      echo "$pkg" >> "$AX_penv_trimmed_requirements"
    fi
  done < "$AX_penv_requirements"
  if $do_tasks; then
    inc_tasks VAR_tasks "$(wc -l "$AX_penv_trimmed_requirements" | awk '{print $1}')"
  fi
  echo "${revised:1}" > "$AX_penv_requirements"
}

install_required_python_libs() {
  revise_requirements true
  install_python_prerequisites
  if [ -z "$(list AX_penv_common)" ]; then
    install_python_libs "libs"
  else
    # We won't reach here if no development nor runtime environment are selected
    install_python_libs "axeleracommon"
    requested_install "development" && install_python_libs "development"
    requested_install "runtime" && install_python_libs "runtime"
  fi
}

install_penv_system() {
  local var=
  if needed "$STATUS_penv_pip" || needed "$STATUS_penv_requirements"; then
    # Any change to pip or requirements requires full generation of environment
    if ! $VAR_pyenv_set_local; then
      cmd "$VAR_pyenv" local "$AX_penv_python" || error "Failed to switch pyenv to version $AX_penv_python"
      VAR_pyenv_set_local=true
    fi
    progress_info "Create Python $AX_penv_python virtual environment"
    cmd mkdir -p "${AX_HOME}/venvs" || error "Failed to create virtual environment directory"
    cmd pip install pyYAML || error "Failed to install pyYAML"
    local env_tag=$(get_env_tag)
    local versioned_env="${AX_HOME}/venvs/${env_tag}"
    cmd rm -rf "$versioned_env" || error "Failed to remove old virtual environment (try first deleting $versioned_env)"
    cmd python3 -m venv "$versioned_env" || error "Failed to create virtual environment (try first deleting $PYENV_ROOT)"
    cmd rm -rf "$VENV" || error "Failed to remove old virtual environment"
    cmd ln -s ${versioned_env} "$VENV" || error "Failed to create symbolic link to virtual environment"
    progress_info "Activate ${ACTIVATE}"
    if $ARG_dry_run; then
      echo "source ${ACTIVATE}"
    else
      # shellcheck disable=SC1090
      source "${ACTIVATE}"
    fi
    complete_task
    progress_info "Install pip ${AX_penv_pip#==}"
    cmd python3 -m ensurepip --upgrade || echo "Warning: ignoring ensurepip failure"
    cmd python3 -m pip install --disable-pip-version-check --upgrade \""pip$AX_penv_pip"\" || error "Failed to install \"pip$AX_penv_pip\""
    install_pip_extra "setuptools"
    install_pip_extra "wheel"
    complete_task
    install_required_python_libs
    patch_activation_script ${env_tag}
  fi
}

# TODO remove git/ssh when we remove current dependency on git revisions
install_penv_docker() {
  write_to_dockerfile "RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git ssh"
  write_to_dockerfile "COPY $AX_penv_requirements ."
  write_to_dockerfile "RUN python3 -m pip install --disable-pip-version-check --upgrade \"pip$(sanitize_version "$AX_penv_pip")\""
  write_to_dockerfile "RUN python3 -m pip install --disable-pip-version-check --upgrade \"setuptools$(sanitize_version "$AX_penv_setuptools")\""
  write_to_dockerfile "RUN python3 -m pip install --disable-pip-version-check --upgrade \"wheel$(sanitize_version "$AX_penv_wheel")\""
  write_to_dockerfile "RUN python3 -m pip install --no-cache-dir --no-deps -r \"$(basename "$AX_penv_requirements")\" $(pip_extra_index_url)"
}

install_needed_python_components() {
  # Install components needed to generate
  # a set of requirements
  install_pyenv_to_system
  configure_pyenv "$(is_dry_arg)"
  install_python_to_pyenv
  configure_pyenv_python "$(is_dry_arg)"
  install_pipenv_to_system
}

install_development() {
  # Install all development environment components
  local list=$(list "AX_penv_python_src_dependencies")
  local var=
  if [ ! -f "$AX_penv_requirements" ] || ! streq "$AX_development_where" "$STR_docker"; then
    install_needed_python_components
  fi
  if [ ! -f "$AX_penv_requirements" ]; then
    gen_pipfile
    pipfile_to_requirements
  fi
  if streq "$AX_development_where" "$STR_docker"; then
    write_to_dockerfile "RUN apt update && apt-get install -y --no-install-recommends wget curl ca-certificates\\"
    for var in $list; do
      write_to_dockerfile " ${!var}\\"
    done
    write_to_dockerfile " && cd /usr/src &&\\"
    write_to_dockerfile " wget $AX_penv_python_src &&\\"
    write_to_dockerfile " tar xzf Python-$AX_penv_python.tgz &&\\"
    write_to_dockerfile " cd Python-$AX_penv_python &&\\"
    write_to_dockerfile " ./configure --enable-optimizations &&\\"
    write_to_dockerfile " make install &&\\"
    write_to_dockerfile " (cd \$(dirname \$(which python3)) && ln -s python3 python) &&\\"
    write_to_dockerfile " rm -rf /var/lib/apt/lists/\*"
    install_docker_packages "AX_dependencies"
    install_penv_docker
  else
    install_system_packages_with_apt "AX_dependencies"
    install_penv_system
  fi
}

set_status_component() {
  local cmp="$1_component"
  eval "$cmp"=\"$STR_ok\"
  set_status_system_libs "$1_dependencies" "$cmp"
  set_status_system_libs "$1_libs" "$cmp"
}

install_component() {
  local where="$1_where"
  if streq "${!where}" "$STR_docker"; then
    install_docker_packages "$1_dependencies"
    install_docker_packages "$1_libs"
  else
    install_system_packages_with_apt "$1_dependencies"
    install_system_packages_with_apt "$1_libs"
  fi
}

install_docker_system() {
  install_system_packages_with_apt "AX_docker_system_dependencies"
}

install_docker_libs() {
  # Regenerate Python requirements
  local list=$(list "AX_docker_libs")
  local var=
  for var in $list; do
    VAR_docker_apt="$VAR_docker_apt ${!var}"
  done
}

docker_postinstall() {
  # Ensure Docker post-installation steps have been performed
  local docker_users=
  if ! docker_users=$(getent group docker); then
    progress_info "Create docker group"
    cmd sudo groupadd docker || error "Failed to create docker group"
  fi
  if [[ ! "$docker_users" =~ .*:$USER ]]; then
    progress_info "Add user $USER as member of docker group"
    cmd sudo usermod -aG docker "$USER" || error "Failed to add user '$USER' to docker group"
  fi
  if [[ ! " ${groups[*]} " =~ " docker " ]] && $ARG_dry_run; then
    # This command cannot be run within the script
    echo "sudo newgrp docker"
  fi

  if ! in_container; then
    if ! systemctl | grep running | grep -q docker.service; then
      progress_info "Start Docker service"
      cmd sudo systemctl --now enable docker
    fi
  fi
}

docker_check() {
  # Check Docker post-installation steps have been performed
  local groups=
  if VAR_docker_users=$(getent group docker); then
    if [[ ! "$VAR_docker_users" =~ .*[,:]$USER ]]; then
      echo "User is not a member of the docker group"
      echo "First run 'sudo usermod -aG docker $USER'"
      echo "Then log out and log back in to re-evaluate your group membership"
      exit 1
    fi
  else
    echo "The 'docker' group does not exist"
    echo "First try:"
    echo "  sudo groupadd docker"
    echo "  sudo usermod -aG docker $USER"
    echo
    echo "Then log out and log back in to re-evaluate your group membership"
    echo "or run the following command to activate group changes in a new shell:"
    echo "  newgrp docker"
    echo
    exit 1
  fi
  #TODO: check that there are no issues running doker commands with 'exec ag docker'
  #IFS=", " read -r -a groups <<< "$(groups)"
  #if is_called_from_docker_launcher && [[ ! " ${groups[*]} " =~ " docker " ]]; then
  #  echo "Log out and log back in to re-evaluate your group membership"
  #  echo "or run the following command to activate group changes in a new shell:"
  #  echo "  newgrp docker"
  #  echo
  #  exit 1
  #fi
  if ! in_container; then
    # With bash as an init process inside containder systemd isn't available
    if ! systemctl | grep running | grep -q docker.service; then
      echo "Docker service not running"
      echo "First run 'sudo systemctl --now enable docker'"
      exit 1
    fi
  fi
}

find_repo_for_lib() {
  # If YAML contains repo for $1, return repo
  local list=$(list_of_dict "AX_repositories")
  local var_i=
  local var_j=
  local packages=
  local repo_list=
  local found=false
  for var_i in $list; do
    packages="${var_i}_packages"
    packages=$(list "$packages")
    repo_list="${var_i}_list"
    repo_list=${!repo_list}
    for var_j in $packages; do
      if streq "${!var_j}" "$1"; then
        echo "$var_i"
        break
      fi
    done
  done
  echo ""
}

set_status_system_libs() {
  # Check whether packages are available for install
  # If a package is missing/outdated then add the
  # repos from the YAML config if they exist. It is
  # assumed the correct version of that package is
  # provided by the repo listed in the YAML config,
  # so that repo can be safely installed without
  # duplication if the package is missing.
  # If the config file gives a version number, then a
  # comparison is made to see if the installed version
  # matches. If not the package is marked for "upgrade".
  # The "upgrade" is not perfect. In any case, we uninstall
  # the current version since we know it doesn't match.
  # Then, if an exact version is specified, we use the
  # limited <pkg>=<version> syntax of apt-get to attempt
  # install, otherwise we just install the version in the
  # supplied repo, which will usually be what we want.
  # TODO: A corner case exists if a repo was previously
  # added but 'apt update' not performed
  # 1: Variable name with list of libs
  # 2: COMPONENT variable with combined status
  local list=$(list "$1")
  local packages=${2//component/packages}
  local repos=${2//component/repos}
  local tasks=${2//component/tasks}
  local where=${2//component/where}
  local var=
  local status=
  local installed_repos=
  local recommended_repos=
  local pkg=
  local add_repo=
  dpkg_info=`dpkg -l | grep -E "^.i" | awk ' {print $2} '` # get installed packages
  for var in $list; do
    status=${var//AX/STATUS}
    IFS=" " read -r -a pkg <<< "${!var}"
    installed_repos=$(apt-cache policy "${pkg[0]}" | grep http | awk '{ print $2 }' | sort | uniq)
    recommended_repos=$(find_repo_for_lib "${pkg[0]}")
    add_repo=false
    # Check status of lib
    if streq "${!where}" "$STR_docker"; then
      if needed "$STATUS_container"; then
        # Install into container
        is_set "$recommended_repos" && add_repo=true
        eval "$status"=\""$STR_docker_needed"\"
        eval "$2"=\""$STR_docker_needed"\"
        eval "$packages"=\(true\)
      else
        eval "$status"=\""$STR_ok"\"
      fi
    elif is_dpkg_installed "${pkg[0]}" "${pkg[1]}" "${pkg[2]}" "$dpkg_info"; then
      # Don't add repos if package already installed
      add_repo=false
      eval "$status"=\"$STR_ok\"
    elif is_dpkg_installed "${pkg[0]}"; then
      # Installed but out of date
      is_set "$recommended_repos" && add_repo=true
      eval "$status"=\""$STR_upgrade_needed"\"
      eval "$2"=\""$STR_needed"\"
      eval "$packages"=\(true\)
      inc_tasks "$tasks"
    else
      # Not installed
      is_set "$recommended_repos" && add_repo=true
      eval "$status"=\""$STR_needed"\"
      eval "$2"=\""$STR_needed"\"
      eval "$packages"=\(true\)
      inc_tasks "$tasks"
    fi
    # Warn about possibly missing packages
    if [[ "${pkg[0]}" == *.deb ]]; then
      if [ ! -f "${pkg[0]}" ]; then
        eval "$status"=\""$STR_unavailable"\"
      fi
    elif ! arg_docker && ! is_set "$installed_repos" && ! is_set "$recommended_repos"; then
      eval "$status"=\""$STR_unavailable"\"
      warn_defer "Cannot find package ${pkg[0]} in any repository"
    fi
    # Add repo to component variable
    if $add_repo; then
      if [[ ! "${!repos}" == *$recommended_repos* ]]; then
        eval "$repos"=\"\$"$repos" "$recommended_repos"\"
      fi
    fi
  done
}

set_status_all() {
  # Set status of all components
  for var in "${!AX_@}"; do
    declare -g "${var//AX/STATUS}"="$STR_unset"
  done
  set_status_os
  arg_docker_or_file && set_status_docker
  $ARG_gen_requirements && set_status_gen_requirements
  ($ARG_no_development && $ARG_no_runtime) || set_status_development
  $ARG_no_runtime || set_status_component "AX_runtime"
  $ARG_no_driver || set_status_component "AX_driver"
}

response_is_yes() {
  # Request yes/no response to $1
  local is_yes=true
  local yn=
  $ARG_YES && is_yes=true
  $VAR_post_table || print_newline
  while true; do
    $ARG_YES && break;
    if is_set "$2"; then
      read -rp "$1 (y/n/h): " yn
    else
      read -rp "$1 (y/n): " yn
    fi
    case $yn in
      [Yy]*)
        break
        ;;
      [Nn]*)
        is_yes=false
        break
        ;;
      [Hh]*)
        echo
        echo "$2"
        echo
        ;;
      * )
        ;;
    esac
  done
  if $is_yes; then
    true
  else
    false
  fi
}

inc_tasks() {
  # 1: Variable
  # 2: Optional number of tasks
  eval "$1"=\($(($1+${2:-1}))\)
}

complete_task() {
  # 1: optional number of tasks
  VAR_i=$((VAR_i+${1:-1}))
}

pip_extra_index_url() {
  # Get pip --extra-index-url value
  local repo=$(list_of_dict "AX_penv_repositories")
  local extra=
  local var=
  local name=
  local ssl=
  local url=
  local pip_extra_args=
  local default=
  local requirement=
  for var in $repo; do
    default="${var}_default"
    url="${var}_url"
    pip_extra_args="${var}_pip_extra_args"
    if ! streq "${!default}" "True"; then
      extra="$extra --extra-index-url=\"${!url}\" ${!pip_extra_args}"
    fi
  done
  echo "${extra:1}"
}

gen_pipfile() {
  # Regenerate Python requirements
  local libs="$(list "AX_penv_.\+_libs")"
  local repo=$(list_of_dict "AX_penv_repositories")
  local var=
  local name=
  local ssl=
  local url=
  local default=
  local requirement=
  if $ARG_dry_run; then
    echo "./$_self --gen-pipfile"
    return
  fi
  progress_info "Generate Pipfile from YAML requirements"
  echo "# Auto-generated Pipfile" > Pipfile
  echo >> Pipfile
  for var in $repo; do
    name="${var}_name"
    name=${!name}
    url="${var}_url"
    url=${!url}
    ssl="${var}_ssl"
    ssl=${!ssl}
    ssl="${ssl,,}"
    {
      echo "[[source]]"
      echo "name = \"${name}\""
      echo "url = \"${url}\""
      echo "verify_ssl = ${ssl}"
      echo
    } >> Pipfile
  done
  echo "[packages]" >> Pipfile
  for var in $libs; do
    requirement=${var//AX_penv_*_libs_/}
    requirement=${requirement#*_}
    if [[ "${!var}" =~ ^.*.whl[[:space:]]*\|.*$ ]]; then
      progress_info "Ignore local wheel ${!var}"
    elif [[ "${!var}" == *.whl* ]]; then
      error "Malformed local wheel requirement '${!var}': prefix with ' | name==version'"
    elif streq "${!var}" "*"; then
      echo "$requirement = \"${!var}\"" >> "Pipfile"
    elif [[ "${!var}" == $'{'*$'}' ]]; then
      echo "$requirement = ${!var}" >> "Pipfile"
    else
      echo "$requirement = \"==${!var}\"" >> "Pipfile"
    fi
  done
  if [ ! -z "${AX_penv_setuptools}" ]; then
    echo "setuptools = \"$(sanitize_version "$AX_penv_setuptools")\"" >> "Pipfile"
  fi
  {
    echo
    echo "[requires]"
    echo "python_version = \"$AX_penv_python\""
  } >> Pipfile
  if ! streq "$VAR_used_cuda_runtime" "cpu"; then
    revise_pipfile_for_compat_cuda "torch" "$VAR_used_cuda_runtime"
    revise_pipfile_for_compat_cuda "torchvision" "$VAR_used_cuda_runtime"
  fi
  complete_task
}

pipfile_to_requirements() {
  # Generate Python requirements file from Pipfile
  local requirements=
  local absolute=
  local relative=
  local resolved=
  local out=
  local dirname=
  local libs=$(list "AX_penv_libs")
  local var=
  local var_x=
  local var_y=
  if $ARG_dry_run; then
    echo "python3 -m pipenv --bare --rm"
    echo "rm -f Pipfile.lock"
    echo "python3 -m pipenv --bare lock"
    echo "PIP_IGNORE_INSTALLED=1 python3 -m pipenv --bare sync"
    echo "python3 -m pipenv run pip freeze > \"$AX_penv_requirements\""
    return
  fi
  progress_info "Generate Pipfile.lock"
  if python3 -m pipenv --venv &> /dev/null; then
    python3 -m pipenv --bare --rm || error "Failed to remove existing environment"
  fi
  cmd_rm Pipfile.lock || error "Failed to remove Pipfile.lock"
  if ! python3 -m pipenv --bare lock 2>&1; then
    #https://github.com/pypa/virtualenv/issues/1875
    out=$(python3 -m pipenv --bare lock 2>&1)
    if [[ "$out" == *"pipenv.exceptions.VirtualenvCreationException"* ]]; then
      echo
      echo "First try: "
      echo "  python3 -m pip uninstall virtualenv"
      echo "  python3 -m pip uninstall pipenv"
      echo "  python3 -m pip install --user pipenv"
      echo
      exit 1
    fi
    error "Failed to lock Pipenv.lock"
  fi
  complete_task
  progress_info "Install packages from Pipenv.lock"
  PIP_IGNORE_INSTALLED=1 python3 -m pipenv --bare sync || error "Failed to install packages from Pipenv.lock"
  complete_task
  # Pipenv cannot deal with dependencies in local wheels
  # Install with pip as post processing operation
  for var in $libs; do
    if [[ "${!var}" =~ ^.*.whl[[:space:]]*\|.*$ ]]; then
      var=$(urldecode "${!var}")
      progress_info "Use pip to install local wheel ${var%\|*}"
      python3 -m pipenv --bare run pip -q install "${var%\|*}" 1>/dev/null || error "Failed to install ${var%\|*} with pip"
    fi
  done
  progress_info "Freeze requirements to $AX_penv_requirements"
  dirname=$(dirname "$AX_penv_requirements")
  if ! streq "$dirname" "."; then
    cmd mkdir -p "$dirname" || error "Failed to mkdir $dirname"
  fi
  if ! python3 -m pipenv run pip freeze > "$AX_penv_requirements"; then
    rm -f "$AX_penv_requirements"
    error "Failed to generate $AX_penv_requirements"
  fi

  # Ensure only one OpenCV package installed
  requirements=$(<"$AX_penv_requirements")
  if [[ "$requirements" =~ opencv-contrib-python-headless==.*$ ]] &&
     [[ "$requirements" =~ opencv-python==.*$ ]]; then
    # opencv-contrib-python-headless and opencv-python = opencv-contrib-python
    requirements=$(echo "$requirements" | sed '/opencv-contrib-python-headless/d')
    requirements=${requirements//opencv-python==/opencv-contrib-python==}
  fi
  if [[ "$requirements" =~ opencv-contrib-python==.*$ ]]; then
    requirements=$(echo "$requirements" | sed '/opencv-contrib-python-headless/d')
    requirements=$(echo "$requirements" | sed '/opencv-python/d')
    requirements=$(echo "$requirements" | sed '/opencv-python-headless/d')
  fi
  if [[ "$requirements" =~ opencv-python==.*$ ]]; then
    requirements=$(echo "$requirements" | sed '/opencv-python-headless/d')
  fi

  # Convert absolute paths to relative
  while IFS="" read -r line || [ -n "$line" ] ; do
    absolute=$(echo "${line}" | grep -Eo "file:///.*$")
    absolute=$(urldecode "$absolute")
    if is_set "$absolute"; then
      relative=${absolute:7}
      relative=$(realpath --relative-to="$SELF_DIR" "$relative")
      resolved=$resolved$'\n'./$relative
    else
      resolved=$resolved$'\n'$line
    fi
  done <<< "$requirements"
  resolved=${resolved:1}
  # Add comments to local wheels including package name/version
  # (needed by installer when diffing requirements with environment)
  for var in $libs; do
    if [[ "${!var}" =~ ^.*.whl[[:space:]]*\|.*$ ]]; then
      var=${!var}
      var_x=$(echo "${var%\|*}" | xargs)
      var_y=$(echo "${var##*\|}" | xargs)
      resolved=${resolved//$var_x/$var_x \# $var_y}
    fi
  done
  echo "$resolved" > "$AX_penv_requirements"
  revise_requirements
  complete_task
  reload_python_requirements
  inc_tasks VAR_tasks "$(wc -l <<< "$VAR_requirements"  | awk NF)"
  VAR_tasks_unknown_requirements=false
}

patch_activation_script() {
  local script=$(cat "${ACTIVATE}" 2>/dev/null)
  local prefix=$(cat .activate_prefix)
  local destruct=$(cat .activate_destruct)
  progress_info "Patch ${ACTIVATE} with code to set additional environment variables"
  if $ARG_dry_run; then
    echo "./$_self --patch-activation > \"${ACTIVATE}\""
  elif [[ "$script" == *"# Patched"* ]]; then
    error "${ACTIVATE} script has already been patched"
  else
    AX_envs=$(print_envs_to_source AX_runtime_envs)
    prefix="${prefix//(.\/install.sh/(cd $PWD && .\/install.sh}"
    prefix="${prefix//-f install.sh/-f $PWD\/install.sh}"
    prefix="${prefix//install.sh:/$PWD\/install.sh:}"
    prefix="${prefix/_AX_envs=/declare -a _AX_envs=(${AX_envs})}"
    if is_set "$1"; then
      script="${script//($1)/(venv)}"
    fi
    script="${script//unset -f deactivate/$destruct}"
    script="$prefix"$'\n\n'"$script"$'\n'"# Patched"
    echo "$script" > "${ACTIVATE}"
  fi
  complete_task
}

http_download() {
  local url=$1
  local path=$2
  local code=
  local dir=
  if ! test -f "$path"; then
    code=$(curl -s -o /dev/null -I -w "%{http_code}" "$url")
    if [ "$code" -eq "200" ]; then
      echo "Download local artifact: $path"
      dir="$(dirname "$path")"
      if [ ! -d "$dir" ]; then
        if [ -L "$dir" ]; then
          warn "$dir: is an invalid symbolic link - not downloading to it"
          return 0
        else
          cmd mkdir -p "$dir" || error "Failed to mkdir $dir"
        fi
      fi
      if ! cmd curl -# "$url" -o "$path"; then
        cmd rm -f "$path"
        warn "$path: Download failed"
      fi
    else
      warn "Failed to download $url: Error $code"
    fi
  fi
}

check_home_folder_link() {
  local home_folder="${AX_HOME}/$1"
  local warning=
  if [ -L "$1" ]; then
    if [ ! "$(readlink "$1")" -ef "$home_folder" ]; then
      warning="$1: is already a symbolic link to $(readlink "$1")"
    fi
  elif [ -e "$1" ]; then
    warning="$1: exists but is not a symbolic link"
  else
    cmd mkdir -p ${home_folder} || error "Failed to create directory $home_folder"
    cmd ln -sfn "$home_folder" "$1" || error "Failed to create symbolic link $1"
  fi
  if is_set "$warning"; then
    warn "$warning"
    warn "This will be used as-is instead of creating a symbolic link to $home_folder"
  fi
}

download_extras() {
  check_home_folder_link "$1"
  local urls=$(list_of_dict "AX_$1")
  local url=
  local base=
  local decoded=
  for url in ${urls}; do
    url="${url}_url"
    base=$(basename "${!url}")
    decoded=$(urldecode "$base")
    http_download "${!url}" "$1/$decoded"
  done
}

prepare_arm_platform() {
  # This attempts to address known problems when installing on a fresh image
  # replace .cn with .com as these have been known to fail
  cmd "sudo sed -i 's|http://mirrors.ustc.edu.cn/ubuntu-ports|http://ports.ubuntu.com/ubuntu-ports|g' /etc/apt/sources.list" || true
  # unhold held packages as they mess up the installation
  cmd "(apt-mark showhold | xargs -r sudo apt-mark unhold) &> /dev/null" || true
  # create a possibly missing symlink needed for metis-dkms
  local kernel=$(uname -r)
  cmd sudo ln -sfn  /usr/src/linux-headers-${kernel} /lib/modules/${kernel}/build || warn "Failed to create symlink /lib/modules/${kernel}/build"
}


_set_env() {
  # strip escapes from $ expressions and evaluate RHS before assigning to LHS
  local x=${2//\\\$/\$}
  x=`echo $x`
  eval "$1=\"$x\""
  export "$1"
}

docker_create_groups() {
  local grp=
  for grp in $*; do
    GROUP_ID=$(cut -d: -f3 <<< $(getent group ${grp}))
    if [ ! -z "$GROUP_ID" ]; then
      GROUP_ID="-g $GROUP_ID"
    fi
    write_to_dockerfile "RUN groupadd $GROUP_ID ${grp}"
  done
}

# ******************************** MAIN ********************************

# Ensure all installer dependencies are satisfied

if ! which dpkg-query &> /dev/null; then
  # Needed to check all other installation requirements
  echo "First run 'sudo apt-get install dpkg'"
  exit 1
fi

if ! which sudo &> /dev/null || ! sudo -vn &> /dev/null; then
  sudo true || error "First install and/or set up sudo permission"
fi

# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
    "--all")               set -- "$@" "-a" ;;
    "--development")       set -- "$@" "-d" ;;
    "--user")              set -- "$@" "-u" ;;
    "--token")             set -- "$@" "-t" ;;
    "--no-development")    set -- "$@" "-D" ;;
    "--runtime")           set -- "$@" "-r" ;;
    "--no-runtime")        set -- "$@" "-R" ;;
    "--driver")            set -- "$@" "-p" ;;
    "--no-driver")         set -- "$@" "-P" ;;
    "--gen-requirements")  set -- "$@" "-g" ;;
    "--gen-pipfile")       set -- "$@" "-i" ;;
    "--patch-activation")  set -- "$@" "-b" ;;
    "--dry-run")           set -- "$@" "-n" ;;
    "--docker")            set -- "$@" "-k" ;;
    "--gen-dockerfile")    set -- "$@" "-K" ;;
    "--print-container")   set -- "$@" "-c" ;;
    "--yes")               set -- "$@" "-y" ;;
    "--YES")               set -- "$@" "-Y" ;;
    "--status")            set -- "$@" "-s" ;;
    "--activate-env")      set -- "$@" "-e" ;;
    "--artifacts")         set -- "$@" "-f" ;;
    "--media")             set -- "$@" "-m" ;;
    "--quiet")             set -- "$@" "-q" ;;
    "--verbose")           set -- "$@" "-v" ;;
    "--allow-as-sudo")     set -- "$@" "-S" ;;
    "--help")              set -- "$@" "-h" ;;
    *)                     set -- "$@" "$arg"
  esac
done

# Parse command-line options
while getopts ":adDrRpPgibnkKcyYsefmqvShu:t:" opt; do
  case $opt in
    a )
      ARG_all=true
      check_no_filegen_arg "all"
      ;;
    d )
      ARG_development=true
      check_no_filegen_arg "development"
      check_arg_or_no_arg "development"
      ;;
    D )
      ARG_no_development=true
      check_arg_or_no_arg "development"
      ;;
    u )
      ARG_user="$OPTARG"
      ;;
    t )
      ARG_token="$OPTARG"
      ;;
    r )
      ARG_runtime=true
      check_no_filegen_arg "runtime"
      check_arg_or_no_arg "runtime"
      ;;
    R )
      ARG_no_runtime=true
      check_arg_or_no_arg "runtime"
      ;;
    p )
      ARG_driver=true
      check_no_filegen_arg "driver"
      check_arg_or_no_arg "driver"
      ;;
    P )
      ARG_no_driver=true
      check_arg_or_no_arg "driver"
      ;;
    g )
      ARG_gen_requirements=true
      check_no_component_install "gen-requirements"
      check_no_sudo "gen-requirements"
      check_no_docker "gen-requirements"
      ;;
    i )
      ARG_gen_pipfile=true
      check_no_component_install "gen-pipfile"
      check_no_sudo "gen-pipfile"
      check_no_docker "gen-pipfile"
      ;;
    b )
      ARG_patch_activation=true
      check_no_component_install "patch-activation"
      check_no_sudo "patch-activation"
      check_no_docker "patch-activation"
      ;;
    n )
      ARG_dry_run=true
      ;;
    k )
      ARG_docker=true
      ARG_docker_system=true
      check_no_filegen_arg "docker"
      check_no_sudo "docker"
      ;;
    K )
      ARG_gen_dockerfile=true
      check_no_filegen_arg "gen-dockerfile"
      check_no_docker "gen-dockerfile"
      check_no_sudo "gen-dockerfile"
      ;;
    c )
      ARG_print_container=true
      ;;
    y )
      ARG_yes=true
      ;;
    Y )
      ARG_yes=true
      ARG_YES=true
      ;;
    s )
      ARG_status=true
      check_no_component_install "status"
      check_no_filegen_arg "status"
      check_no_docker "status"
      ;;
    e )
      ARG_activate_env=true
      check_no_component_install "activate-env"
      check_no_filegen_arg "activate-env"
      check_no_docker "activate-env"
      ;;
    f )
      ARG_artifacts=true
      ;;
    m )
      ARG_media=true
      ;;
    q )
      ARG_quiet=true
      check_only_one_of_arg "quiet" "verbose"
      ;;
    v )
      ARG_verbose=true
      check_only_one_of_arg "quiet" "verbose"
      ;;
    S )
      ARG_allow_as_sudo=true
      ;;
    h )
      echo "Usage:"
      echo "  $_self [options]"
      echo
      echo "Install the Voyager SDK"
      echo
      echo "  -a  --all                install all components"
      echo "  -d  --development        install Python development environment and runtime (--no-development supported)"
      echo "  -u  --user <user>        email address used for the registered Axelera AI account (see also docs/tutorials/install.md)"
      echo "  -t  --token <token>      token generated using the registered Axelera AI account (see also docs/tutorials/install.md)"
      echo "  -r  --runtime            install runtime libraries (--no-runtime supported)"
      echo "  -p  --driver             install PCIe/dkms driver (--no-driver supported)"
      echo "      --artifacts          download local artifacts needed by the installer"
      echo "      --media              download local media needed for tests/demos"
      echo "      --dry-run            print installation commands instead of executing"
      echo "      --docker             install into a Docker container"
      echo "      --gen-dockerfile     generate Dockerfile"
      echo "      --gen-pipfile        generate Pipfile from YAML"
      echo "      --gen-requirements   generate Python requirements from YAML"
      echo "      --patch-activation   patch Pyenv activation script with additional functionality"
      echo "      --print-container    print container name/tag and exit"
      echo "      --status             check install status of components"
      echo "      --yes                answer yes to most installation questions (--YES for all)"
      echo "      --quiet              display less output"
      echo "      --verbose            display more output"
      echo "      --allow-as-sudo      allow installer to be run as sudo - not recommended"
      echo "  -h, --help               display this help and exit"
      echo
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      echo "Try '$_self --help' for a list of supported options"
      exit 1
      ;;
    * )
      echo "Invalid option: -$OPTARG" >&2
      echo "Try '$_self --help' for a list of supported options"
      exit 1
  esac
done

shift $((OPTIND-1))

exit_if_error

# USER not always set (in containers?)
if [ -z $USER ]; then
  USER=$(whoami)
fi

# check if running sudo - or might have been previously
if [[ $(id -u) -eq 0 ]] && [[ -n $SUDO_USER ]]; then
  if $ARG_allow_as_sudo; then
    warn "*** Running as root - this may not work properly - run as a regular user if possible ***"
  else
    error "$_self is not intended to be run as root - it uses sudo where required. Please install as a regular user or use --allow-as-sudo to override this."
  fi
elif [[ $USER != "root" ]]; then
  # check if some key files are root-owned and warn
  if [[ $(find ~/.pyenv -maxdepth 2 -user root 2>/dev/null) ]] || [[ $(stat -c %U .python-version 2>/dev/null) == "root" ]]; then
    warn "Some files in ~/.pyenv and/or .python-version are owned by root, which may have been run as root previously. This may cause issues with the installer."
  fi
fi

if [[ ${SYS_arch} == "arm64" ]] && ! in_container; then
  prepare_arm_platform
fi

# Packages required by the installer itself
installer_apt_requirement "python3"
installer_apt_requirement "python3-pip"
installer_apt_requirement "lsb-release"
installer_apt_requirement "apt-utils"
installer_apt_requirement "coreutils"
installer_apt_requirement "git"
installer_apt_requirement "curl"
installer_apt_requirement "ca-certificates"
installer_apt_requirement "gnupg"
installer_apt_requirement "cmake"
installer_apt_requirement "xauth"
# ensure required apt packages are installed before attempting any pip packages
check_installer_requirements_met

installer_pip_requirement "pyYAML"
check_installer_requirements_met

if $ARG_all; then
  set_arg_if_not_unset "development"
  set_arg_if_not_unset "driver"
  set_arg_if_not_unset "runtime"
fi

# If --gen-requirements specified then no
# other components should be installed
if $ARG_gen_requirements; then
  ARG_no_development=true
  ARG_no_runtime=true
  ARG_no_driver=true
fi

# If --gen-dockerfile specified then ignore
# --driver (even if it is specified)
if $ARG_gen_dockerfile; then
  ARG_driver=false
  ARG_no_driver=true
fi

# Get configuration settings
get_config_from_yaml "$SYS_config"
AX_penv_trimmed_requirements=$(dirname "$AX_penv_requirements")/trimmed-$(basename "$AX_penv_requirements")

# DEBUG
#for var in "${!AX_@}"; do
# echo $var=${!var}
#done

# When installing with --docker option we need to know the
# container name, so disable interactive installation
# If --docker specified then require either development environment
# with runtime or runtime alone (set up here without prompting)
if arg_docker_or_file; then
  if $ARG_development && $ARG_no_runtime; then
    echo "Docker development environment must contain runtime"
    exit 1
  elif ! $ARG_development && $ARG_runtime; then
    # probably wants runtime only
    warn "Defaulting to --no-development (specify --no-development to disable this warning)"
    ARG_no_development=true
  elif ! $ARG_no_development; then
    ARG_development=true
    ARG_runtime=true
  elif $ARG_no_runtime; then
    echo "No docker components selected"
    exit 1
  else
    ARG_runtime=true
  fi
fi

# If user wants --runtime-only then use a simple container without CUDA
# Otherwise select CUDA compatible runtime if availble, or default to CPU
VAR_installed_cuda_runtime=$(installed_cuda_runtime)
if arg_docker_or_file && $ARG_no_development; then
  STATUS_installed_cuda_runtime=$STR_unused
  VAR_used_container=$AX_docker_base
  VAR_target_container=$AX_docker_target_runtime
else
  STATUS_installed_cuda_runtime=$STR_ok
  VAR_used_cuda_runtime=$(selected_cuda_runtime "$VAR_installed_cuda_runtime" "compat")
  is_set "$VAR_used_cuda_runtime" || VAR_used_cuda_runtime="cpu"
  VAR_used_container=$(selected_cuda_runtime "$VAR_installed_cuda_runtime" "docker_base")
  VAR_target_container=$(selected_cuda_runtime "$VAR_installed_cuda_runtime" "docker_target")
  add_to_list "AX_docker_system_dependencies" "AX_docker_system_cuda_dependencies"
fi

# If can't find a container, default to base container
if ! is_set "$VAR_used_container"; then
  VAR_used_container=$AX_docker_base
  VAR_target_container=$AX_docker_target_base
fi

# Check base container exists
if ! is_set "$VAR_used_container"; then
  STATUS_base_docker=$STR_unsupported
else
  STATUS_base_docker=$STR_ok
fi

# Load Python requirements after selecting a CUDA runtime
load_python_requirements

# We can only calculate development container tags/hash once the Pyhon
# requirements exists
if arg_docker && ! $ARG_no_development && ! is_set "$VAR_requirements"; then
  echo "First run './${_self} --gen-requirements' (needed to calculate Docker container tag)"
  exit 1
fi

# Get Docker container tag after loading Python requirements
if [ -f "$AX_penv_requirements" -a -z "$VAR_target_container_tag" ]; then
  VAR_target_container_tag=$(get_tag_from_yaml "$SYS_config" "$AX_penv_requirements" "docker_hash")
fi

# Determine location of each component and the status of any existing container
if arg_docker_or_file; then
  AX_development_where=$STR_docker
  AX_runtime_where=$STR_docker
  STATUS_container=$STR_needed
  if [ -z $USER ]; then
    USER=$(whoami)
  fi
  if arg_docker && [[ -n "$(which docker)" ]]; then
    docker_check
    if exec sg "docker" "docker image ls" | grep -q "^$VAR_target_container.*$VAR_target_container_tag.*$"; then
      STATUS_container=$STR_ok
    fi
  fi
else
  AX_development_where=$STR_system
  AX_runtime_where=$STR_system
fi

# Make sure we have a reference to where we expect pyenv
# to be installed (for printing, may be installed later)
if ! is_set "$PYENV_ROOT"; then
  export PYENV_ROOT="$HOME/.pyenv"
else
  VAR_pyenv_root_env=true
fi

VAR_pyenv="$PYENV_ROOT/bin/pyenv"
if streq "$(which pyenv)" "$VAR_pyenv"; then
  VAR_pyenv_in_path=true
fi

if ! streq "$AX_development_where" "$STR_docker"; then
  configure_pyenv
fi

# create and symlink a data folder if not already present
check_home_folder_link "data"

# we don't currently have any "artifacts" but keeping this in case we do in the future
if $ARG_artifacts; then
  download_extras "artifacts"
fi

if $ARG_media; then
  download_extras "media"
fi

# Determine installation status of all components
set_status_all

# We now have sufficient information for the launch script
if is_called_from_docker_launcher; then
  return 0
fi

if $ARG_status; then
  print_config
  exit 0
fi


if $ARG_print_container; then
  echo "$VAR_target_container" "$VAR_target_container_tag"
  exit 0
fi

# User requested activation of the environment
if $ARG_activate_env; then
  print_envs_to_source "AX_runtime_envs"
  exit 0
fi

# User requested Pipfile generation
if $ARG_gen_pipfile; then
  VAR_tasks="1"
  gen_pipfile
  exit 0
fi

# User requested patching activation script
if $ARG_patch_activation; then
  VAR_tasks="1"
  patch_activation_script $(get_env_tag)
  exit 0
fi

# Other installation tasks cannot happen within a virtual environment
if is_set "$VIRTUAL_ENV"; then
  echo "First run 'deactivate' and then re-run the installer"
  exit 1
fi

# Print installation status
print_config
print_deferred_warnings

# With --docker option, if the required container is not
# already on the system, we need to install it
if needed "$STATUS_container"; then
  VAR_install=true
fi

arg_docker && resolve_option "docker_system" "Install Docker system dependencies"
if ! arg_docker; then
  resolve_option "development" "Install Python development environment"
elif needed "$STATUS_container"; then
  resolve_option "development" "Install Python development environment and runtime"
fi

if ! requested_install "development"; then
  # Only the development component can result in an
  # unknown number of installation tasks
  VAR_tasks_unknown_requirements=false
fi

resolve_option "runtime" "Install runtime libraries"
resolve_option "driver" "Install PCIe driver"

if $ARG_gen_requirements; then
  resolve_option "gen_requirements" "Generate requirements"
fi

AX_GROUPS="video render messagebus"

if arg_docker; then
  update_repo_list "$AX_docker_repos" "$STR_docker"
else
  # check if user should be added to any groups
  VAR_groups=""
  for group in $AX_GROUPS; do
    if getent group $group &> /dev/null && (! groups $USER | grep -q "\b$group\b"); then
      VAR_groups="$VAR_groups $group"
    fi
  done
  if is_set "$VAR_groups"; then
    print_config_line "Need to add user to groups:$VAR_groups"
    print_config_header
  fi
fi

# Final confirmation step before running installation tasks
if $VAR_install || is_set $VAR_groups; then
  if $ARG_dry_run; then
    msg="Proceed with displaying commands to execute"
  elif $ARG_gen_requirements; then
    msg="Proceed with generating requirements"
  elif $ARG_gen_dockerfile; then
    msg="Proceed with generating Dockerfile"
  else
    msg="Proceed with installation"
  fi
  if ! response_is_yes "$msg"; then
    echo "No action performed"
    print_newline
    exit 1
  fi
elif streq "$AX_development_component" "$STR_ok"; then
  print_next_steps "No pending installation tasks"
  exit 0
else
  exit 0
fi

if is_set $VAR_groups; then
  added=false
  for group in $VAR_groups; do
    if cmd sudo usermod -a -G $group $USER; then
      added=true
    else
      echo "Warning: failed to add $USER to group $group"
    fi
  done
  if $added && ! $ARG_dry_run; then
    echo "Log out and log back in to apply group changes"
  fi
  if ! $VAR_install && streq "$AX_development_component" "$STR_ok"; then
    print_next_steps "Installation complete"
    exit 0
  fi
fi

# Update task count
if $VAR_any_system_packages; then
  inc_tasks VAR_tasks 2 # apt check and update
fi
if arg_docker; then
  inc_tasks VAR_tasks 1 # docker build
fi

# Prompt for sudo password if it will be needed later
if ! $ARG_dry_run && $VAR_any_system_packages && ! sudo -vn 2>/dev/null; then
  echo "Checking for 'sudo' permission to install packages"
  sudo true || error "Terminated due to permission error"
fi

print_newline

# Start defining a new container
if needed "$STATUS_container"; then
  if $ARG_dry_run; then
    if $ARG_no_development; then
      echo "./$_self --gen-dockerfile --no-development"
    else
      echo "./$_self --gen-dockerfile"
    fi
  else
    cmd_rm Dockerfile || error "Failed to remove Dockerfile"
    write_to_dockerfile "FROM $VAR_used_container"
    write_to_dockerfile "ARG DEBIAN_FRONTEND=noninteractive"
    write_to_dockerfile "ENV DISPLAY=:0"
    is_set "$AX_docker_term" && write_to_dockerfile "ENV TERM $AX_docker_term"
    write_to_dockerfile "WORKDIR /tmp"
    print_envs_to_source "AX_runtime_envs"
    install_docker_libs
    VAR_uid="$(id -u)"
    write_to_dockerfile "RUN useradd --badname -m -u $VAR_uid $USER"

    docker_create_groups render axelera

    write_to_dockerfile "RUN usermod -a -G sudo,video,render,axelera $USER"
    write_to_dockerfile "RUN echo '$USER:$USER' | chpasswd"

    # create a volume to replace the host's .local
    write_to_dockerfile "RUN mkdir -p $HOME/.local && chown -R $USER:$USER $HOME/.local"
    write_to_dockerfile "VOLUME $HOME/.local"
  fi
fi

# Add repositories
if is_set "$VAR_system_repositories"; then
  apt_check_system
  add_apt_repositories "VAR_system_repositories" "$STR_system"
  apt_update_system
fi
if is_set "$VAR_docker_repositories"; then
  add_apt_repositories "VAR_docker_repositories" "$STR_docker"
fi

# Install requested components
if $ARG_gen_requirements; then
  install_needed_python_components
  gen_pipfile
  pipfile_to_requirements
fi

requested_install "docker_system" && install_docker_system
if requested_install "runtime" || requested_install "development"; then
  install_development
fi
requested_install "runtime" && install_component "AX_runtime"
requested_install "driver" && install_component "AX_driver"

# do post inst tasks

# According to the sudo documentation:
#
#   sudo will read each file in /etc/sudoers.d, skipping file names that
#   end in ‘~’ or contain a ‘.’ character to avoid causing problems with
#   package manager or editor temporary/backup files.
#
# Hence, in order to make this work, we need to strip the dot from the
# username when determining the filename inside /etc/sudoers.d.
USERNAME_WITHOUT_DOTS=$(echo "$USER"|sed 's/\.//g')
superpower="echo $USERNAME_WITHOUT_DOTS ALL=\\(ALL\\) NOPASSWD: ALL > /etc/sudoers.d/$USERNAME_WITHOUT_DOTS || echo Warning: failed to give $USER sudo permissions"
superpower="bash -c \"$superpower\""

if ! needed "$STATUS_container"; then
  if ! $ARG_dry_run && ! $ARG_gen_requirements; then
    # add USER to sudoers
    cmd "sudo $superpower"

    # remaining tasks need active env
    # shellcheck disable=SC1090
    if [ -n "${_OLD_VIRTUAL_PATH:-}" ] ; then
      # undo any previous venv activation
      deactivate || true
    fi
    source "${ACTIVATE}"
    echo building operators
    cmd "make clobber-libs && make operators" || warn "Failed to make operators"

    # rescan pcie and reload firmware
    if [ -d "/sys/bus/pci" ]; then
      echo refreshing pcie and firmware
      # on some platforms this can take a couple of tries
      (axdevice --refresh &> /dev/null && axdevice --refresh) || warn "Failed to refresh pcie and firmware"
    fi
  fi
else
  # Remove unnecessary Docker files and build a container
  DOK="DOCKER_BUILDKIT=1 docker build --compress -f Dockerfile -t \"$VAR_target_container:$VAR_target_container_tag\" ."
  DOK="${DOK} --network=\"host\""

  if $ARG_verbose; then
    DOK="${DOK} --progress=plain"
  fi
  write_to_dockerfile "RUN apt-get update && apt-get autoremove -y && apt-get install -y --no-install-recommends\\"
  for var in $VAR_docker_apt; do
    write_to_dockerfile " $var\\"
  done
  write_to_dockerfile " && rm -rf /var/lib/apt/lists/\*"
  # now do some post install things in the container
  # add USER to sudoers
  write_to_dockerfile "RUN $superpower"
  if $ARG_gen_dockerfile; then
    $ARG_dry_run || echo "Wrote output to Dockerfile"
  elif $ARG_dry_run; then
    docker_postinstall
    echo "$DOK"
  else
    docker_postinstall
    progress_info "Build container $VAR_target_container:$VAR_target_container_tag "
    IFS=", " read -r -a GROUPS <<< "$(groups)"
    if [[ ! " ${GROUPS[*]} " =~ " docker " ]]; then
      exec sg "docker" "$DOK" || error "Docker build command failed"
      echo "Newgroup instruction"
    else
      eval "$DOK" || error "Docker build command failed"
    fi
    complete_task
  fi
fi

if $VAR_any_errors; then
  echo "Installation complete, but with unresolved issues (see above)"
  exit 1
fi

if ! $ARG_dry_run && ! $ARG_gen_dockerfile && ! $ARG_gen_requirements; then
  print_next_steps "Installation complete"
fi
