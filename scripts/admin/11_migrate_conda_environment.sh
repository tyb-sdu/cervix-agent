#!/usr/bin/env bash
set -euo pipefail

project_root="/data2/lxj/projects/CervixAgent"
conda_executable="/data2/lxj/miniconda3/bin/conda"
source_environment="/data2/lxj/miniconda3/envs/cervixagent"
target_environment="${project_root}/.envs/core"
backup_environment="${project_root}/tmp/core_external_environment_backup_20260724"
archive_path="${project_root}/tmp/core_environment_20260724.tar.gz"
record_root="${project_root}/configs/environments"

if [[ ! -x "${conda_executable}" ]]; then
    printf 'Missing conda executable: %s\n' "${conda_executable}" >&2
    exit 2
fi
if [[ ! -x "${source_environment}/bin/python" ]]; then
    printf 'Missing source environment: %s\n' "${source_environment}" >&2
    exit 2
fi
if [[ -e "${target_environment}" ]]; then
    printf 'Target environment already exists: %s\n' "${target_environment}" >&2
    exit 2
fi
if [[ -e "${backup_environment}" ]]; then
    printf 'Backup environment already exists: %s\n' "${backup_environment}" >&2
    exit 2
fi

mkdir -p \
    "${record_root}" \
    "$(dirname "${target_environment}")" \
    "$(dirname "${backup_environment}")"
"${conda_executable}" list --explicit --prefix "${source_environment}" \
    > "${record_root}/core_before_migration_explicit.txt"

# Avoid accepting new Anaconda channel Terms of Service merely to relocate an
# existing environment. conda-pack is installed from PyPI and performs an
# offline, relocatable copy of the already installed packages.
"${source_environment}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    conda-pack
"${source_environment}/bin/conda-pack" \
    --prefix "${source_environment}" \
    --output "${archive_path}" \
    --ignore-editable-packages \
    --force
mkdir -p "${target_environment}"
tar -xzf "${archive_path}" -C "${target_environment}"
"${target_environment}/bin/conda-unpack"
"${target_environment}/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --no-deps \
    --editable "${project_root}"

"${target_environment}/bin/python" -c \
    'import rdkit, sys; print(sys.version); print(rdkit.__version__)'
"${target_environment}/bin/python" -m unittest discover \
    -s "${project_root}/tests" \
    -v \
    > "${record_root}/core_migration_tests.log" 2>&1

"${target_environment}/bin/python" -m pip freeze \
    > "${record_root}/core_project_local_pip_freeze.txt"

source_packages_hash="$(
    sha256sum "${record_root}/core_before_migration_explicit.txt" |
        awk '{print $1}'
)"
target_packages_hash="$(
    sha256sum "${record_root}/core_project_local_pip_freeze.txt" |
        awk '{print $1}'
)"

# Keep the exact original environment as a temporary rollback copy, but move
# it under the CervixAgent project so no project file remains mixed into the
# shared Conda environment directory.
mv -- "${source_environment}" "${backup_environment}"
if [[ -e "${source_environment}" ]] || [[ ! -d "${backup_environment}" ]]; then
    printf 'Old external environment relocation failed.\n' >&2
    exit 4
fi
rm -f -- "${archive_path}"

printf '%s  %s\n' \
    "${target_packages_hash}" \
    "core_project_local_pip_freeze.txt" \
    > "${record_root}/core_project_local_pip_freeze.txt.sha256"

printf 'TARGET_ENVIRONMENT=%s\n' "${target_environment}"
printf 'PYTHON=%s\n' "$("${target_environment}/bin/python" --version 2>&1)"
printf 'RDKIT=%s\n' "$("${target_environment}/bin/python" -c 'import rdkit; print(rdkit.__version__)')"
printf 'UNIT_TESTS=passed\n'
printf 'SOURCE_CONDA_MANIFEST_SHA256=%s\n' "${source_packages_hash}"
printf 'TARGET_PIP_FREEZE_SHA256=%s\n' "${target_packages_hash}"
printf 'OLD_EXTERNAL_ENVIRONMENT_MOVED_UNDER_PROJECT=true\n'
printf 'ROLLBACK_COPY=%s\n' "${backup_environment}"
