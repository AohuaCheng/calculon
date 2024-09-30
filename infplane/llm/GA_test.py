import datetime
import gzip
import logging
import multiprocessing as mp
import psutil
import os

import infplane
from infplane.util import pick, arg_true_false_all
from infplane.llm import *

import pygad

class GAExecution(infplane.CommandLine):
  NAME = 'llm-GA-test'
  ALIASES = ['GA']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      GAExecution.NAME, aliases=GAExecution.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=GAExecution.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-m', '--mbs-break', action='store_true',
                    help='Search across MBS and break earlier when possible')
    sp.add_argument('-t', '--top-n', type=int, default=1,
                    help='Number of best outputs')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')
    sp.add_argument('--no-tp-overlap', action='store_true',
                    help='Don\'t allow TP overlap')
    sp.add_argument('--no-dp-overlap', action='store_true',
                    help='Don\'t allow DP overlap')
    sp.add_argument('-training', default='true',
                    help='train or test')

  @staticmethod
  def run_command(logger, args):
    assert args.top_n > 0, 'top-n must be > 0'

    app = Llm.Application(infplane.io.read_json_file(args.application))
    syst = System(infplane.io.read_json_file(args.system))

    num_run = len(list(Llm.get_all_tensor_parallelisms(args.num_procs, app.hidden, app.attn_heads)))
    print("GA run iterations ", num_run)
    for tp in Llm.get_all_tensor_parallelisms(args.num_procs, app.hidden, app.attn_heads):
        results = GAExecution.GA_search(args.layers, args.num_procs, args.max_batch_size, args.datatype,
              app, syst, tp, args.fused_activation, not args.no_tp_overlap, not args.no_dp_overlap, args.training)

    print(results)
    # # Runs parallel searches
    # start_time = datetime.datetime.now()
    # with mp.Pool(args.cpus) as pool:
    #   searches = pool.starmap(GAExecution.GA_search, params)
    # end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    # best = []
    # exe_count = 0
    # good_exe_count = 0
    # bad_exe_count = 0
    # for cbest in searches:
    #   best = GAExecution.update_list(best, cbest, args.top_n)
      # exe_count += ec
      # good_exe_count += gec
      # bad_exe_count += bec

    # logger.info(f'Total executions: {exe_count}')
    # logger.info(f'Good executions: {good_exe_count}')
    # logger.info(f'Bad executions: {bad_exe_count}')
    # calc_rate = exe_count / (end_time - start_time).total_seconds()
    # logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    # if args.debug:
    #   return 0

    # if len(best) == 0:
    #   if not args.noneok:
    #     logger.fatal('No acceptable configurations found :(')
    #     return -1
    #   else:
    #     logger.info('No acceptable configurations found :(')
    # else:
    #   logger.info(f'Best sample rate: {best[0][0]}')

    # output = {}
    # for index, run in enumerate(best):
    #   _, execution, stats = run
    #   output[index] = {
    #     'execution': execution,
    #     'stats': stats
    #   }

    # if infplane.io.is_json_extension(args.output):
    #   logger.info(f'Output: {args.output}')
    #   infplane.io.write_json_file(output, args.output)
    # elif args.output.endswith('.csv') or args.output.endswith('.csv.gz'):
    #   logger.info(f'Output: {args.output}')
    #   exe_keys = list(output[0]['execution'].keys())
    #   stats_keys = list(output[0]['stats'].keys())
    #   opener = gzip.open if args.output.endswith('.gz') else open
    #   with opener(args.output, 'wb') as fd:
    #     fd.write(bytes(f',{",".join(exe_keys)},{",".join(stats_keys)}\n',
    #                    'utf-8'))
    #     for index in sorted(output.keys()):
    #       fd.write(bytes(f'{index}', 'utf-8'))
    #       for exe_key in exe_keys:
    #         fd.write(bytes(f',{output[index]["execution"][exe_key]}', 'utf-8'))
    #       for stats_key in stats_keys:
    #         fd.write(bytes(f',{output[index]["stats"][stats_key]}', 'utf-8'))
    #       fd.write(bytes('\n', 'utf-8'))
    # else:
    #   assert False, f'Unknown file type: {args.output}'
    return 0

  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    if data_par > max_batch_size:
      return None
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par


  @staticmethod
  def GA_search(layers, num_procs, max_batch_size, datatype,
              app, syst, tp, fused_acts,
              allow_tp_overlap, allow_dp_overlap, training):
    num_nets = syst.num_networks
    has_mem2 = syst.mem2.capacity > 0
    param_list = [
                list(Llm.get_all_pipeline_parallelisms(num_procs, tp, app.num_blocks)), # pp
                range(3), # for activation_recompute in ['full', 'attn_only', 'none']
                [0,1], # for optimizer_sharding in pick(dp>1, [True, False], [False]):
                range(3), # for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                [0,1], # if can_redo else [0],  # seq_par_ag_redo
                [0,1], # if dp > 1 and allow_dp_overlap else [0],  # data_par_overlap
                range(3) if tp > 1 and allow_tp_overlap else [0],  # tensor_par_overlap
                [0,1] if has_mem2 else [0],  # weight_offload
                [0,1], #[0] if activation_recompute == 'full' or not has_mem2 else [0,1],  # activations_offload
                [0,1] if has_mem2 else [0],  # optimizer_offload
                np.array(fused_acts).astype(int),  # fused_act
                # list(Llm.get_valid_microbatch_sizes(app.seq_size, tp, dp, batch_size, pp)),  # microbatch_size
                range(num_nets) if tp > 1 else [0],  # tn 
                range(num_nets), #if pp > 1 else [0],  # pn
                range(num_nets), #if dp > 1 else [0] # dn
            ]
    
    def fitness_func(ga_instance, solution, solution_idx):
      return get_sample_rate(solution, return_json=False)
    
    def get_sample_rate(solution, return_json=True):
      pp, activation_recompute, optimizer_sharding, tensor_par_comm_type, seq_par_ag_redo, data_par_overlap, tensor_par_overlap, weight_offload, activations_offload, optimizer_offload, fused_act, tn, pn, dn = solution
      seq_par_ag_redo, data_par_overlap, weight_offload, activations_offload, optimizer_offload, fused_act, tn, pn, dn = int(seq_par_ag_redo), int(data_par_overlap), int(weight_offload), int(activations_offload), int(optimizer_offload), int(fused_act), int(tn), int(pn), int(dn)
      activation_recompute_list = ['full', 'attn_only', 'none']
      activation_recompute = activation_recompute_list[int(activation_recompute)]
      tensor_par_overlap_list = ['none', 'ring', 'pipe']
      tensor_par_overlap = tensor_par_overlap_list[int(tensor_par_overlap)]
      tensor_par_comm_type_list = ['ar', 'p2p_rs_ag', 'rs_ag']
      tensor_par_comm_type = tensor_par_comm_type_list[int(tensor_par_comm_type)]

      pp, optimizer_sharding = int(pp), int(optimizer_sharding)
      dp = Llm.get_data_parallelism(num_procs, tp, pp)
      if dn>0 and dp<=1:
        return -1
      if optimizer_sharding>0 and dp<=1:
        return -1 
      if pn>0 and pp<=1:
        return -1
      if data_par_overlap==1 and not (dp > 1 and allow_dp_overlap):
        return -1
      can_redo = Llm.can_redo_ag(tensor_par_comm_type,
                                activation_recompute)
      if seq_par_ag_redo==1 and not can_redo:
        return -1
      if (activation_recompute == 'full' or not has_mem2) and activations_offload==1:
        return -1
      
      results = [(-1,None)]
      for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
        batch_size = GAExecution.get_batch_size(dp, max_batch_size)
        if batch_size is None:
          continue
        for microbatch_size in list(Llm.get_valid_microbatch_sizes(app.seq_size, tp, dp, batch_size, pp)):
    
          exe_json = {
            'num_procs': num_procs,
            'tensor_par': tp,
            'pipeline_par': pp,
            'data_par': dp,
            'tensor_par_net': tn,
            'pipeline_par_net': pn,
            'data_par_net': dn,
            'batch_size': batch_size,
            'microbatch_size': microbatch_size,
            'datatype': datatype,
            'fused_activation': fused_act,
            'attention_type': 'multihead',
            'activation_recompute': activation_recompute,
            'pipeline_interleaving': ppint,
            'optimizer_sharding': optimizer_sharding,
            'tensor_par_comm_type': tensor_par_comm_type,
            'tensor_par_overlap': tensor_par_overlap,
            'seq_par_ag_redo': seq_par_ag_redo,
            'data_par_overlap': data_par_overlap,
            'weight_offload': weight_offload,
            'activations_offload': activations_offload,
            'optimizer_offload': optimizer_offload,
            'training': training
          }
          
          try:
            logger = logging.Logger('sub')
            model = Llm(app, logger)
            model.compile(
              syst,
              Llm.Execution.from_json(exe_json))
            model.run(syst)
            stats = model.get_stats_json(layers)
            # good_exe_count += 1
            # curr = (stats['sample_rate'], exe_json, stats)
            results.append((stats['sample_rate'],exe_json))
            
          except Llm.Error as ex:
            logger = logging.getLogger()
            logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
            # bad_exe_count += 1
            results.append((-1,None))
      if return_json:
        return max(results, key=lambda x: x[0])
      else:
        return max(results, key=lambda x: x[0])[0]
      
    num_generations = 100 # Number of generations.
    num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

    sol_per_pop = 20 # Number of solutions in the population.
    num_genes = len(param_list)

    global last_fitness 
    last_fitness = 0
    def on_generation(ga_instance):
        # global last_fitness
        # print(f"Generation = {ga_instance.generations_completed}")
        # print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
        # print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
        # last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        pass

    ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          fitness_func=fitness_func,
                          on_generation=on_generation,
                          gene_space=param_list)

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    # ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    if ga_instance.best_solution_generation != -1:
        print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
    
    return get_sample_rate(solution)

infplane.CommandLine.register(GAExecution)