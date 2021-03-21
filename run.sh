#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

cd "$(dirname "$0")"

IMAGE=python:3.8.1-buster


docker run \
    --rm \
    --interactive \
    --tty \
    --name "${PWD##*/}" \
    --volume "${PWD}:/repo" \
    "${IMAGE}" \
    /repo/_run.sh
