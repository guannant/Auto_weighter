import json
import numpy as np
from ..utils.NSGA_related import get_pareto_front_indices, initialize_gaussian_pool, select_parent_indices, get_operator_menu, get_bounds_and_constraints, get_index_mapping_note
from ..utils.CALPHAD_related import batch_objective_eval
from ..agents.chatbox import openai_chat_completion  
from network import build_ea_langgraph_merged

def run_optimization(
    n_var: int = 22,
    n_obj: int = 22,  # kept for completeness in case you use it elsewhere
    pool_size: int = 20,
    max_generations: int = 50,
    seed: int = 0,
    weights_path: str = "ESPEI_run_file/Pytorch_MLP_CV/weights.json",
    lower_bound: float = 1e-2,
    upper_bound: float = 1000.0,
    init_std: float = 0.8,
    budget: int | None = None,
    recursion_limit: int = 10_000,
    most_recent: int = 50,
) -> None:
    """
    Main entry point for running the LLM-agentic evolutionary optimization.

    Parameters
    ----------
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives (kept for completeness).
    pool_size : int
        Number of candidates in the population.
    max_generations : int
        Maximum number of generations to run.
    seed : int
        Random seed for reproducibility.
    weights_path : str
        Path to the initial weights.json file.
    lower_bound : float
        Lower bound for all parameters.
    upper_bound : float
        Upper bound for all parameters.
    init_std : float
        Standard deviation (or scale) used to initialize the Gaussian pool.
    budget : int | None
        Evaluation budget; defaults to n_obj if not provided.
    recursion_limit : int
        LangGraph recursion limit.
    most_recent : int
        Number of most recent candidates to keep / track (if used downstream).
    """

    rng = np.random.default_rng(seed=seed)

    # Bounds
    lower = np.full(n_var, lower_bound, dtype=float)
    upper = np.full(n_var, upper_bound, dtype=float)
    bounds = (lower, upper)

    # LLM interface
    llm = openai_chat_completion

    # Load initial parameter vector from weights.json
    with open(weights_path, "r") as f:
        init_params_dict = json.load(f)
    init_params = np.array(list(init_params_dict.values()), dtype=float)

    # Initialize population
    param_pool = initialize_gaussian_pool(
        rng=rng,
        init_params=init_params,
        pool_size=pool_size,
        std=init_std,
        bounds=bounds,
    )

    # Initial objective evaluation
    objectives = batch_objective_eval(param_pool, 0)

    # Select parents
    selected_idx = select_parent_indices(rng, objectives, pool_size)
    parent_pool = param_pool[selected_idx]
    parent_objectives = objectives[selected_idx]

    # Operator menu & metadata
    operator_menu = get_operator_menu()
    bounds_and_constraints = get_bounds_and_constraints(bounds)
    index_mapping_note = get_index_mapping_note(n_var)

    history = [
        {
            "objectives": parent_objectives,
            "parent_pool": parent_pool,
        }
    ]

    # Default budget: use number of objectives if not provided
    if budget is None:
        budget = n_obj

    # Initial state for the LangGraph workflow
    init_state = {
        "parent_pool": parent_pool,
        "parent_objectives": parent_objectives,
        "llm": llm,
        "operator_menu": operator_menu,
        "bounds_and_constraints": bounds_and_constraints,
        "index_mapping_note": index_mapping_note,
        "history": history,
        "bounds": bounds,
        "pool_size": pool_size,
        "generation": 0,
        "budget": budget,
        "max_generations": max_generations,
        "all_para": parent_pool,
        "all_obj": parent_objectives,
        "rng": rng,
        "eps_vals": 0.0,
        "most_recent": most_recent,
    }

    workflow = build_ea_langgraph_merged()

    print("Starting optimization...")
    step = 1

    # Stream over the workflow execution
    for event in workflow.stream(init_state, config={"recursion_limit": recursion_limit}):
        # Each event is a dict {node_name: state}
        node_name, node_state = next(iter(event.items()))

        if node_name != "EvalAndSurvivor":
            continue

        print(f"----- Generation {step} -----")

        eps = node_state.get("new_epsilon", node_state.get("eps_vals", 0.0))
        all_obj = node_state["all_obj"]

        pareto_idx = get_pareto_front_indices(all_obj, epsilon=eps)
        pareto_objs = all_obj[pareto_idx]

        # Average RMSE per candidate, then mean & std across Pareto set
        candidate_means = np.mean(pareto_objs, axis=1)
        avg_err_mean = float(np.mean(candidate_means))
        avg_err_std = float(np.std(candidate_means))

        print(f"  Epsilon: {eps:.4f}")
        print(f"  Avg Pareto RMS (mean over objectives): {avg_err_mean:.4f}")
        print(f"  Std Pareto RMS (mean over objectives): {avg_err_std:.4f}")

        step += 1

    print("Optimization finished.")


if __name__ == "__main__":
    run_optimization()
