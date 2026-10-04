#!/usr/bin/env bash
# Exercise 8: run Compute Sanitizer tools (run from repo root after `make`).
set -x
compute-sanitizer --tool memcheck  ./vector_gridstride 128 256 1000
compute-sanitizer --tool memcheck  ./reduce_sum
compute-sanitizer --tool racecheck ./reduce_sum
compute-sanitizer --tool synccheck ./reduce_sum
compute-sanitizer --tool memcheck  ./matrix_mul_tiled
compute-sanitizer --tool racecheck ./matrix_mul_tiled
compute-sanitizer --tool synccheck ./matrix_mul_tiled
# Experiment 1: missing bound check -> invalid global read
compute-sanitizer --tool memcheck  ./vector_add_bug
# Experiment 2: missing first barrier -> shared-memory hazards
compute-sanitizer --tool racecheck ./reduce_sum_bug
# Fixed version should be clean
compute-sanitizer --tool memcheck  ./vector_add_fixed
