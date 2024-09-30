#!/bin/bash

set -e

export PYTHONPATH=.

# CLI interface infrastructure
echo -e "### Testing top level --help"
./bin/infplane --help > /dev/null
commands=$(./bin/infplane --help | head -n 2 | tail -n 1 | tr '{' ' ' | tr '}' ' ' | tr ',' ' ')
for command in $commands; do
    if [ $command == 'v' ] || [ $command == 'version' ]; then
	echo -e "### Testing \"$command\""
	./bin/infplane $command
    else
	echo -e "### Testing \"$command\" --help"
	./bin/infplane $command --help > /dev/null
    fi
done
echo -e "\n\n"

# Model size calculations
echo -e "### Testing llm-parameter-calculator"
for model in models/*json; do
    ./bin/infplane llm-parameter-calculator -a 15 $model
done
echo -e "\n\n"

# Model tests
echo -e "### Testing llm"
for model in models/*json; do
    echo $model
    ./bin/infplane llm $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json - > /dev/null
    ./bin/infplane llm $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json /tmp/infplane_stats.json -p /tmp/infplane_peers.json
done
echo -e "\n\n"

# Llm validation
echo -e "### Testing llm-validation"
./bin/infplane lv -v
echo -e "\n\n"

# Llm optimal execution
echo -e "### Testing llm-optimal-execution (float16) (using -f)"
./bin/infplane loe models/turing-530B.json 5128 2520 float16 systems/h100_80g_nvl8.json /tmp/infplane_530B_fp16.json -t 3 -f False --no-tp-overlap
echo -e "\n"

echo -e "### Testing llm-optimal-execution (float8) (using -m)"
./bin/infplane loe models/turing-530B.json 5128 2520 float8 systems/h100_80g_nvl8.json /tmp/infplane_530B_fp8.csv.gz -t 10 -m
echo -e "\n\n"

# Llm all executions
echo -e "### Testing llm-all-executions (float8)"
./bin/infplane lae models/turing-530B.json 5128 2520 float8 systems/h100_80g_nvl8.json /tmp/infplane_530B_fp8_all.csv.gz
echo -e "\n\n"

