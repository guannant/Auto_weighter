import os
import re
import json
import yaml
import subprocess
from collections import defaultdict
import numpy as np
from espei.database import Database, load_datasets
from espei.data_generator import DataGenerator
from espei.utils import recursive_glob, unpack_piecewise
from espei.parameter_selection import database_symbols_to_fit



def group_by_comps(input_dict):
    groups = defaultdict(list)
    for key, val_list in input_dict.items():
        # Extract the comps part (the content within parentheses after 'comps:')
        match = re.search(r"comps:\s*(\([^)]+\))", key)
        if not match:
            print("find no match")
            continue  # skip keys that do not contain the comps part
        comps_str = match.group(1)
        
        # Remove the colon and anything inside curly braces.
        # For example, transform "LAVES_C15: {X_MG: 0.31592}" to "LAVES_C15"
        cleaned = re.sub(r":\s*\{[^}]*\}", "", comps_str)
        # Remove extra spaces around commas and parentheses:
        cleaned = re.sub(r"\s*,\s*", ",", cleaned)
        cleaned = re.sub(r"\(\s*", "(", cleaned)
        cleaned = re.sub(r"\s*\)", ")", cleaned)
        
        # Use the cleaned comps string as the group key
        groups[cleaned].extend(val_list)
    
    # Compute the average for each group
    result = {}
    for comps, values in groups.items():
        avg = sum(values) / len(values) if values else None
        result[comps] = avg
        
    return result

def plot_grouped_bar(data_list):
    # Group values by key
    grouped_data = {}
    for d in data_list:
        for key, value in d.items():
            grouped_data.setdefault(key, []).append(abs(value))
    
    # Sort keys for consistent x-axis ordering
    keys = sorted(grouped_data.keys())    
    feed_for_gpt = {}
    # Plot each group's bars
    for i, key in enumerate(keys):
        values = grouped_data[key]
        feed_for_gpt[key] = np.average(values)
    return feed_for_gpt

def plot_bar_dict(data_dict):
    # Extract keys and values (assuming each value list has one element)
    keys = list(data_dict.keys())
    abe = [abs((data_dict[k][0][1] - data_dict[k][0][0])) * 100 for k in keys]
    feed_for_gpt ={k:v/100 for k, v in zip(keys, abe)}
    return feed_for_gpt
title ={
    "CUMG2_HMR": 20.42921352219551,
    "CUMG2_SMR": 0.528,
    "FCC_A1_HM_MIX": 13.410089581522925,
    "FCC_A1_SM_MIX": 0.528,
    "HCP_A3_HM_MIX": 8.467937527171069,
    "LAVES_C15_HMR": 26.362360143472927,
    "LAVES_C15_SMR": 0.528,
    "LAVES_C15_HM_MIX": 12.247749925325591,
    "LIQUID_HM_MIX": 12.593890860385642,
    "LIQUID_SM_MIX": 0.528,
    "(LAVES_C15)": 489.24809999999997,
    "(LAVES_C15,FCC_A1)": 43.486763079751725,
    "(LAVES_C15,LIQUID)": 59.47870989563999,
    "(LAVES_C15,CUMG2)": 5.489296573964274,
    "(LIQUID,LAVES_C15)": 0.6808378299754795,
    "(HCP_A3,CUMG2)": 17.7269585724,
    "(LIQUID,HCP_A3)": 478.8,
    "(LIQUID,CUMG2)": 48.083841263001595,
    "(LIQUID,FCC_A1)": 175.2315980625,
    "(FCC_A1,LAVES_C15)": 82.35009441792002,
    "(FCC_A1,LIQUID)": 12.411224504756625,
    "(FCC_A1)": 197.27386015759902
}
def process_output(out):
    zpf_errors = [] # by each dataset
    for item in out[1]:
        zpf_errors.append(group_by_comps(item))
    zpf_value = plot_grouped_bar(zpf_errors)
    thermoc_val = plot_bar_dict(out[0])
    merged_dict = {**thermoc_val,**zpf_value}
    ordered_keys = list(title.keys())
    # catA_keys = ordered_keys[:10]
    # catB_keys = ordered_keys[10:]
    # catA_error = {k: merged_dict[k] for k in catA_keys}
    # catB_error = {k: merged_dict[k] for k in catB_keys}
    # ordered_dict = {k: merged_dict[k] for k in ordered_keys}
    ordered_out = [merged_dict[k] for k in ordered_keys]
    return  ordered_out

phase_models_path = open('ESPEI_run_file/phase_models.json')
dataset_path = 'ESPEI_run_file/input-data_entropyIncl'
tdb_initial_path = 'ESPEI_run_file/Cu-Mg-generated.tdb'
phase_models = json.load(phase_models_path)
dbf = Database(tdb_initial_path)
datasets = load_datasets(sorted(recursive_glob(dataset_path, '*.json')))
data_gen = DataGenerator(dbf, datasets, phase_models)
def eval_output(tbd_file_path):

    dbf = Database(tbd_file_path)
    symbols_to_fit = database_symbols_to_fit(dbf)
    point = np.array([unpack_piecewise(dbf.symbols[s]) for s in symbols_to_fit])
    output = data_gen.eval_grouped_error(point)
    return process_output(output)
def objective_fn(params: np.ndarray, iteration: int, post_fix: int) -> list:

    yaml_path = "ESPEI_run_file/run_mcmc.yaml"
    output_folder = "ESPEI_run_file"
    param_names = list(title.keys())
    params_dict = dict(zip(param_names, params))
    weights_path = os.path.join(output_folder, "Pytorch_MLP_CV", "weights.json")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)  # Ensure subdir exists
    with open(weights_path, "w") as wf:
        json.dump(params_dict, wf, indent=4)
    
    # 1. Read YAML
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 2. Set output_db based on iteration
    base_name = f"LLM_agent_{iteration}_{post_fix}"
    db_path = os.path.join(output_folder, f"{base_name}.tdb")
    trace_path = os.path.join(output_folder, f"{base_name}_trace.npy")
    prob_path = os.path.join(output_folder, f"{base_name}_lnprob.npy")

    data["output"]["output_db"] = db_path
    data["output"]["tracefile"] = trace_path
    data["output"]["probfile"] = prob_path
    # 3. Write YAML back
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)

    # 4. Run espei
    if post_fix < 0:
        try:
                subprocess.run(["espei", "--input", yaml_path], check=True)
        except subprocess.CalledProcessError as e:
                print(f"ESPEI run failed: {e}")
                return None  # or handle as appropriate
    else:
        if not os.path.exists(db_path):
            try:
                subprocess.run(["espei", "--input", yaml_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"ESPEI run failed: {e}")
                return None  # or handle as appropriate
        else:
            print(f"Database {db_path} already exists, skipping ESPEI run.")

    evaluated_errors = eval_output(db_path)
    return evaluated_errors
def batch_objective_eval(parameter_array, iteration_start=0):
    """
    parameter_array: shape (N, D)
    Returns: np.ndarray of objective values, shape (N, M)
      (where M is number of objectives)
    """
    results = []
    for i, params in enumerate(parameter_array):
        # This assumes your objective_fn returns a dict of errors for each objective.
        # If your dict is not sorted, use a consistent key order (e.g., same as 'title')
        evaluated_errors = objective_fn(params, iteration=iteration_start, post_fix=i)
        results.append(evaluated_errors)
    return np.array(results)
