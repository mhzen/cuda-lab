#!/usr/bin/env bash
# Exercise 5: sweep 9 launch configurations, median of 5 runs each.
# Usage: ./scripts/sweep_gridstride.sh   (run from repo root after `make`)
N=1000003
echo "blocks threads median_ms GB/s"
for b in 64 128 256; do
  for t in 64 128 256; do
    med=$(for r in 1 2 3 4 5; do
            ./vector_gridstride $b $t $N | awk '/^blocks=/{split($3,k,"="); print k[2]}'
          done | sort -n | sed -n 3p)
    awk -v b=$b -v t=$t -v m=$med -v n=$N \
      'BEGIN{printf "%d %d %.6f %.2f\n", b, t, m, 12*n/(m*1e-3)/1e9}'
  done
done
