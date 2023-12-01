if [ $# -eq 0 ]

then

diff serial/out.txt parallel/out.txt \
    --old-group-format=$'\nOLD:\n\e[0;31m%<\e[0m' \
    --new-group-format=$'NEW:\n\e[0;31m%>\e[0m\n' \
    --unchanged-group-format=$'SAME:\n\e[0;32m%=\e[0m'

else

diff $1 $2 \
    --old-group-format=$'\nOLD:\n\e[0;31m%<\e[0m' \
    --new-group-format=$'NEW:\n\e[0;31m%>\e[0m\n' \
    --unchanged-group-format=$'SAME:\n\e[0;32m%=\e[0m'

fi
