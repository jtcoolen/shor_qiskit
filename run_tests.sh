#!/usr/bin/env bash
# Run the verification suite.  Every test is a standalone script that asserts
# its own results; a non-zero exit means something regressed.
#
#   ./run_tests.sh fast     unit tests only            (~3 minutes)
#   ./run_tests.sh full     adds end-to-end factoring  (~30 minutes)
#   ./run_tests.sh all      adds N=21 / N=33 and ECDLP  (hours; 8 GB peak RAM)
#   ./run_tests.sh ec       the ECDLP suite only       (~6 minutes)
#   ./run_tests.sh doc      check Part VII matches the benchmark  (instant)
#
# SHOR_EC_FULL=1 widens the ECDLP suite (more primes, more cases, more
# projective representatives).  Default is the narrow sweep.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/shor_qiskit:$PWD/tests:${PYTHONPATH:-}"
PY=${PYTHON:-python3}

FAST="qft_check test_qrom test_tempand test_fixup test_mbu test_unlookup
      test_essentials test_rc test_windowed test_nested test_coset_law
      test_semiclassical test_resources"
E2E="test_rc_l4 test_win_l4 test_acc_mbu test_win_full test_nested_e2e test_win_e2e test_1c"
SLOW="test_precision test_n21 test_win_n21 test_n33"

# --- ECDLP (elliptic-curve discrete log) ------------------------------------
EC="test_ec_classical test_ec_quantum test_ec_arith test_ec_kaliski
    test_ec_pointadd test_ec_shor test_ec_opt106 test_ec_opt1128 test_ec_pbt"

# Part VII of shor-complete.tex is generated from bench/ec_ablation.json; this
# asserts the document still matches what the benchmark measured.
DOC="../bench/ec_check_tex"

case "${1:-fast}" in
  fast) SET="$FAST" ;;
  full) SET="$FAST $E2E" ;;
  all)  SET="$FAST $E2E $SLOW $EC" ;;
  ec)   SET="$EC" ;;
  doc)  SET="" ; "$PY" bench/ec_check_tex.py; exit $? ;;
  *)    echo "usage: $0 [fast|full|all|ec]"; exit 2 ;;
esac

fails=0
for t in $SET; do
  printf '\n=== %s ===\n' "$t"
  if ! "$PY" "tests/$t.py"; then fails=$((fails+1)); echo "FAILED: $t"; fi
done
printf '\n%s\n' "-----------------------------------------"
if [ "$fails" -eq 0 ]; then echo "all suites passed"; else echo "$fails suite(s) FAILED"; fi
exit "$fails"
