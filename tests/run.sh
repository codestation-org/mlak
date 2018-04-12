#! /bin/sh

MODE="report -m"

if [ ${#} -eq 1 ] ; then
	MODE="${1}"
	echo "With mode '${MODE}'."
fi

export TF_CPP_MIN_LOG_LEVEL=3

python3-coverage run --source=. -m unittest discover -v && python3-coverage ${MODE} --omit='tests/*,setup.py,mlak/Visual.py'

