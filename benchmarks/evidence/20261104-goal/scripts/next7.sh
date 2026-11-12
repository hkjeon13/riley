#!/bin/bash
# next7.sh DONE_FILE OLD_SCRIPT NEW_SCRIPT: when DONE_FILE appears, stop OLD_SCRIPT and its servers on GPU 7, start NEW_SCRIPT.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until [ -f "$1" ]; do sleep 5; done; sleep 15
for pat in "[b]ash $2" "[c]ompare.py .* 7 lp" "[r]iley-lp[0-9]* serve .*1833" "[l]oad.py 127.0.0.1:1833"; do
  for p in $(ps -eo pid,cmd | grep -E "$pat" | awk '{print $1}'); do kill $p; done; sleep 2; done
sleep 15
for p in $(ps -eo pid,cmd | grep -E "[r]iley-lp[0-9]* serve .*1833" | awk '{print $1}'); do kill -9 $p; done
sleep 3
until ! curl -s -m 2 127.0.0.1:18330/readyz >/dev/null && ! curl -s -m 2 127.0.0.1:18331/readyz >/dev/null; do sleep 5; done
echo STARTING $3; bash $3
