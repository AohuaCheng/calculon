# Infplane

## Running

Run Infplane like this:
``` sh
$> PYTHONPATH=. ./bin/infplane <args>
```

Infplane is a hierarchical command line. To see the commands it accepts, use `--help` or `-h`:
``` sh
$> PYTHONPATH=. ./bin/infplane -h
```

You can also see how to use any command specifically by using `--help` or `-h` on the command:
``` sh
$> PYTHONPATH=. ./bin/infplane llm -h
```

## LLM Example

Run a single calculation for LLM (~1 sec):
``` sh
$> PYTHONPATH=. ./bin/infplane llm models/megatron-1T.json examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80g.json -
```

Run a system execution optimizer for LLM (~1 min):
``` sh
$> PYTHONPATH=. ./bin/infplane llm-optimal-execution models/turing-530B.json 5128 2520 float16 systems/a100_80g.json output.json -m
```
`opt_exe.json` will contain the optimal way to run Turing-530B across 5128 A100 GPUs.

To store results from all successful runs from the same experiment, run a special system optimizer (~1 min):
``` sh
$> PYTHONPATH=. ./bin/infplane llm-all-executions models/turing-530B.json 5128 2520 float16 systems/a100_80g.json all_output.csv
```

GA algorithm test version (~1 min)
``` sh
$> PYTHONPATH=. ./bin/infplane llm-GA-test models/turing-530B.json 5128 2520 float16 systems/a100_80g.json output.json -m
```
