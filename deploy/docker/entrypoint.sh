#!/bin/sh
# Entrypoint of the distribution image: bundled model presets plus container
# defaults for `riley serve`. Anything it does not recognize goes to `riley`.
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

  serve --preset NAME [riley serve options]
      Download the preset into $MODELS/NAME on first use, then serve it with
      --model-id NAME --inference-profile $PROFILE --bind $BIND
      and a weight limit that fits the checkpoint. Options you pass win.
  serve --model DIR --model-id ID [options]
      Serve a checkpoint you prepared yourself (listens on $BIND).
  download NAME...      Download and verify presets into $MODELS.
  presets               List bundled presets.
  riley-checkpoint ...  Run the checkpoint tool (prepare, verify, download).
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

preset_plan() {
    plan=$PRESETS/$1.json
    [ -f "$plan" ] || die "unknown preset '$1'; run 'presets' to list them"
    echo "$plan"
}

# Sum of every file the manifest lists: an upper bound for --max-weight-bytes.
plan_bytes() {
    sed -n 's/.*"bytes": \([0-9][0-9]*\).*/\1/p' "$1" | awk '{ total += $1 } END { printf "%.0f\n", total }'
}

fetch() {
    plan=$(preset_plan "$1")
    destination=$MODELS/$1
    if [ -f "$destination/riley-checkpoint.json" ]; then
        return
    fi
    if [ -e "$destination" ]; then
        die "$destination exists but holds no finished checkpoint (an interrupted download?); remove it and retry"
    fi
    echo "riley-docker: downloading preset $1 into $destination" >&2
    if [ -n "${HF_TOKEN:-}" ]; then
        riley-checkpoint download --manifest "$plan" --destination "$destination" \
            --allow-network --token-env HF_TOKEN
    else
        riley-checkpoint download --manifest "$plan" --destination "$destination" \
            --allow-network
    fi
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
    preset=
    expect_preset=
    for argument; do
        shift
        if [ -n "$expect_preset" ]; then
            preset=$argument
            expect_preset=
            continue
        fi
        case $argument in
            --preset) expect_preset=1; continue ;;
            --preset=*) preset=${argument#--preset=}; continue ;;
        esac
        set -- "$@" "$argument"
    done
    [ -z "$expect_preset" ] || die "--preset needs a name"

    if [ -n "$preset" ]; then
        has_flag --model "$@" && die "--preset and --model are mutually exclusive"
        plan=$(preset_plan "$preset")
        fetch "$preset"
        set -- --model "$MODELS/$preset" "$@"
        has_flag --model-id "$@" || set -- "$@" --model-id "$preset"
        has_flag --inference-profile "$@" || set -- "$@" --inference-profile "$PROFILE"
        has_flag --max-weight-bytes "$@" || set -- "$@" --max-weight-bytes "$(plan_bytes "$plan")"
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
        [ $# -gt 0 ] || die "download needs at least one preset name"
        for name; do
            fetch "$name"
            echo "riley-docker: $name ready in $MODELS/$name"
        done
        ;;
    serve) serve "$@" ;;
    riley-checkpoint) exec riley-checkpoint "$@" ;;
    *) exec riley "$command" "$@" ;;
esac
