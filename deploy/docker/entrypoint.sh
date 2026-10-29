#!/bin/sh
# Entrypoint of the distribution image: Hub model ids, bundled presets and
# container defaults for `riley serve`. Anything it does not recognize goes to
# `riley`.
set -eu

PRESETS=/opt/riley/presets
MODELS=${RILEY_MODEL_DIR:-/models}
BIND=${RILEY_BIND:-0.0.0.0:8080}
PROFILE=native-bf16-varlen-v1

die() {
    echo "riley-docker: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Riley inference server (OpenAI-compatible HTTP API).

  serve --model MODEL [riley serve options]
      MODEL is a Hugging Face id such as Qwen/Qwen2.5-7B-Instruct (optionally
      @branch, @tag or @commit), a bundled preset name, or a checkpoint
      directory. Hub models are planned from the Hub's own hashes and
      downloaded into $MODELS/hub on first use. Riley then serves with
      --model-id MODEL --inference-profile $PROFILE --bind $BIND
      and a weight limit that fits the checkpoint. Options you pass win.
  serve --preset NAME [options]   Same as --model NAME.
  download MODEL...     Download and verify models into $MODELS without serving.
  presets               List bundled presets (verified on A100).
  riley-checkpoint ...  Run the checkpoint tool (plan, download, prepare, verify).
  --version, --help     Passed to riley; so is any other command.

Gated models need HF_TOKEN in the environment. Presets:
EOF
    list_presets
}

list_presets() {
    for plan in "$PRESETS"/*.json; do
        name=$(basename "$plan" .json)
        source=$(sed -n 's/.*"source_model": "\([^"]*\)".*/\1/p' "$plan")
        printf '  %-18s %s\n' "$name" "$source"
    done
}

# Sum of every file the manifest lists: an upper bound for --max-weight-bytes.
plan_bytes() {
    sed -n 's/.*"bytes": \([0-9][0-9]*\).*/\1/p' "$1" | awk '{ total += $1 } END { printf "%.0f\n", total }'
}

# riley-checkpoint plan/download, with the Hub token when one is set.
hub_checkpoint() {
    if [ -n "${HF_TOKEN:-}" ]; then
        riley-checkpoint "$@" --allow-network --token-env HF_TOKEN
    else
        riley-checkpoint "$@" --allow-network
    fi
}

# Downloads PLAN into DESTINATION unless a finished checkpoint is already there.
fetch_plan() {
    plan=$1
    destination=$2
    if [ -f "$destination/riley-checkpoint.json" ]; then
        return
    fi
    if [ -e "$destination" ]; then
        die "$destination exists but holds no finished checkpoint (an interrupted download?); remove it and retry"
    fi
    echo "riley-docker: downloading into $destination" >&2
    hub_checkpoint download --manifest "$plan" --destination "$destination"
}

is_commit() {
    printf '%s' "$1" | grep -Eq '^[0-9a-f]{40}$'
}

# Resolves a Hub id to a verified local checkpoint in RESOLVED_DIR.
fetch_hub() {
    repository=${1%%@*}
    revision=main
    case $1 in *@*) revision=${1#*@} ;; esac
    root=$MODELS/hub/$(printf '%s' "$repository" | sed 's#/#--#')
    if is_commit "$revision" && [ -f "$root/$revision/riley-checkpoint.json" ]; then
        RESOLVED_DIR=$root/$revision
        return
    fi
    mkdir -p "$root"
    plan=$root/.plan.$$
    rm -f "$plan"
    if ! hub_checkpoint plan --repository "$repository" --revision "$revision" --output "$plan"; then
        rm -f "$plan"
        # Offline restart: fall back to the newest finished download.
        for cached in $(ls -t "$root" 2>/dev/null); do
            if [ -f "$root/$cached/riley-checkpoint.json" ]; then
                echo "riley-docker: Hub unavailable; using cached commit $cached" >&2
                RESOLVED_DIR=$root/$cached
                return
            fi
        done
        die "could not plan $repository@$revision"
    fi
    commit=$(sed -n 's/.*"source_revision": "\([0-9a-f]*\)".*/\1/p' "$plan")
    is_commit "$commit" || die "plan for $repository has no commit"
    fetch_plan "$plan" "$root/$commit"
    rm -f "$plan"
    RESOLVED_DIR=$root/$commit
}

# Sets RESOLVED_DIR and RESOLVED_ID for a directory, preset or Hub id.
resolve() {
    RESOLVED_ID=$1
    case $1 in
        /* | ./* | ../*)
            [ -d "$1" ] || die "$1 is not a directory"
            RESOLVED_DIR=$1
            RESOLVED_ID=$(basename "$1")
            ;;
        */*)
            printf '%s' "$1" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(@[A-Za-z0-9._/-]+)?$' \
                || die "'$1' is not a Hugging Face model id"
            RESOLVED_ID=${1%%@*}
            fetch_hub "$1"
            ;;
        *)
            [ -f "$PRESETS/$1.json" ] || die "unknown preset '$1'; run 'presets', or pass a Hub id such as Qwen/Qwen3-4B"
            RESOLVED_DIR=$MODELS/$1
            fetch_plan "$PRESETS/$1.json" "$RESOLVED_DIR"
            ;;
    esac
}

has_flag() {
    flag=$1
    shift
    for argument; do
        case $argument in
            "$flag" | "$flag"=*) return 0 ;;
        esac
    done
    return 1
}

serve() {
    model=
    expect_model=
    for argument; do
        shift
        if [ -n "$expect_model" ]; then
            [ -z "$model" ] || die "pass one --model or --preset"
            model=$argument
            expect_model=
            continue
        fi
        case $argument in
            --model | --preset) expect_model=1; continue ;;
            --model=* | --preset=*)
                [ -z "$model" ] || die "pass one --model or --preset"
                model=${argument#*=}
                continue
                ;;
        esac
        set -- "$@" "$argument"
    done
    [ -z "$expect_model" ] || die "--model needs a value"
    [ -n "$model" ] || die "serve needs --model (a Hub id, preset or directory)"

    resolve "$model"
    set -- --model "$RESOLVED_DIR" "$@"
    has_flag --model-id "$@" || set -- "$@" --model-id "$RESOLVED_ID"
    has_flag --inference-profile "$@" || set -- "$@" --inference-profile "$PROFILE"
    if [ -f "$RESOLVED_DIR/riley-checkpoint.json" ] && ! has_flag --max-weight-bytes "$@"; then
        set -- "$@" --max-weight-bytes "$(plan_bytes "$RESOLVED_DIR/riley-checkpoint.json")"
    fi
    has_flag --bind "$@" || set -- "$@" --bind "$BIND"
    exec riley serve "$@"
}

command=${1:-help}
[ $# -eq 0 ] || shift
case $command in
    help) usage ;;
    presets) list_presets ;;
    download)
        [ $# -gt 0 ] || die "download needs at least one model"
        for name; do
            resolve "$name"
            echo "riley-docker: $name ready in $RESOLVED_DIR"
        done
        ;;
    serve) serve "$@" ;;
    riley-checkpoint) exec riley-checkpoint "$@" ;;
    *) exec riley "$command" "$@" ;;
esac
