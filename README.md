# cisc372_nbody

Nicholas DiGirolamo

A parallel and serial implementation of a solution to the nbody problem.

Use the following command to compare two output files.

diff --old-group-format=$'\nOLD:\n\e[0;31m%<\e[0m' --new-group-format=$'NEW:\n\e[0;31m%>\e[0m\n' --unchanged-group-format=$'SAME:\n\e[0;32m%=\e[0m' serial/out.txt parallel/out.txt
