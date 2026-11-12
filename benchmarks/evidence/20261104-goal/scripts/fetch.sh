#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102
B=$W/target/release/riley-checkpoint
mkdir -p $W/plans
for r in "$@"; do n=${r##*/}
  [ -d $W/models/$n ] && continue
  $B plan --repository $r --revision main --output $W/plans/$n.json --allow-network && $B download --manifest $W/plans/$n.json --destination $W/models/$n --allow-network && echo "OK $n" || echo "FAIL $n"
done
echo FETCH_DONE
