#! /bin/sh

MODE="report -m"

if [ ${#} -eq 1 ] ; then
	MODE="${1}"
	echo "With mode '${MODE}'."
fi

export TF_CPP_MIN_LOG_LEVEL=3

COVERAGE="python3-coverage"
if which coverage > /dev/null 2>&1 ; then
	COVERAGE="coverage"
fi

${COVERAGE} run --branch --source=. -m unittest discover -v && ${COVERAGE} ${MODE} --omit='tests/*,setup.py,mlak/Visual.py'

