#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
prior="$(dirname "$stage")/w3-rag-prefill-order-retry-20261124.cWPk8nNQ"
(cd "$prior"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        363b6b74595a877213c081c1e04e0f2f4636544c95074d091632fe98f6139856 campaign-dispatch.sha256 \
        05a2b94105a54ead7209a4c0987d0d792cad74fb2a6a356bb394844070c90dfa result.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c campaign-dispatch.sha256 result.sha256 >/dev/null)
cp -- "$prior"/{campaign.sh,configure.py,holdout.py,campaign-dispatch.sha256} .
exec bash campaign.sh
