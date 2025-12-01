from langgraph.graph import StateGraph, START, END
from ..agents.repair import create_llm_condense_repair_agent
from ..agents.diversity import create_llm_diversity_agent
from ..utils.NSGA_related import nsga2_survival, nsga2_tournament_selection
from ..utils.population_summary import summarize_population
from ..utils.NSGA_related import apply_strategy_rank_based
from ..utils.CALPHAD_related import batch_objective_eval
import numpy as np

def condense_repair_agent_node(state):
    # Gather everything needed for LLM-based condense/repair/generation

    llm = state["llm"]
    all_para = state.get("all_para")
    all_obj = state.get("all_obj")
    bounds = state.get("bounds", (None, None))

    # Optionally, get these values from state or set defaults

    # Get population statistics as in your prep_for_debate_node

    summary = summarize_population(all_para, all_obj, n_pca_components=2)

    # Prepare the input dict for the LLM node


    state["summary"] = summary  # Store summary in state for future nodes

    # Create and call the LLM condense/repair agent node
    condense_agent = create_llm_condense_repair_agent(llm, max_retries=10)
    state = condense_agent(state)

    # Unpack new pool and rationales
    condensed_pool = state.get("condensed_pool")
    print(condensed_pool[0])
    rationales = state.get("rationales")
    condensed_pool = np.clip(condensed_pool, bounds[0], bounds[1])

    # objectives_new = batch_objective_eval(condensed_pool) 

    # Update state for next node
    state.update({
        "parent_pool": condensed_pool,
    })
    return state

def diversity_agent_node(state):
    llm = state["llm"]
    diversity_agent = create_llm_diversity_agent(llm, max_retries=10)
    state = diversity_agent(state)

    diverse_pool = state.get("diverse_pool", state["parent_pool"])

    state["parent_pool"] = diverse_pool
    print(f"[Generation {state['generation']}] Diversity agent applied.")
    print(diverse_pool[0])
    return state


def apply_strategy_node(state):
    parent_pool = state["parent_pool"]
    parent_objectives = state["parent_objectives"]
    bounds = state["bounds"]
    rng = state["rng"]
    offsprings = apply_strategy_rank_based(
        rng, parent_pool, parent_objectives, bounds=bounds, mutation_std=0.05
    )
    state["offsprings"] = offsprings
    #state["op_labels"] = op_labels
    return state

def evaluate_and_survivor_node(state):
    # Combine offspring evaluation and survivor selection
    state["generation"] += 1
    offsprings = state["offsprings"]
    offspring_objectives = batch_objective_eval(offsprings, state["generation"])
    parent_pool = state["parent_pool"]
    parent_objectives = state["parent_objectives"]
    rng = state["rng"]

    state["all_para"] = np.vstack(([state["all_para"], state["offsprings"]]))
    state["all_obj"] = np.vstack(([state["all_obj"], offspring_objectives]))
    #state['eps_vals'] = state['eps_vals'] = np.std(state['all_obj'][get_pareto_front_indices(state['all_obj'],epsilon=state['eps_vals'])])
    par_chi_obj = np.vstack([parent_objectives, offspring_objectives])
    par_chi_params = np.vstack([parent_pool, offsprings])
    eps = state.get("new_epsilon", state['eps_vals'])
    survivors, ranks, crowd = nsga2_survival(rng, par_chi_obj, n_survive=parent_pool.shape[0],
                                                epsilon=eps, max_allowed=None,
                                                )
    while len(survivors) == 0:
        eps *= 0.9
        state["new_epsilon"] = eps
        survivors, ranks, crowd = nsga2_survival(rng, par_chi_obj, n_survive=parent_pool.shape[0],
                                                epsilon=eps, max_allowed=None,
                                                )
            
    mating_pool = nsga2_tournament_selection(rng, ranks[survivors],
                                         crowd[survivors],
                                         n_parents=parent_pool.shape[0],
                                         )
    next_indices = survivors[mating_pool]

    new_parent_pool = par_chi_params[next_indices]
    new_parent_objectives = par_chi_obj[next_indices]
    state["parent_pool"] = new_parent_pool
    state["parent_objectives"] = new_parent_objectives
    state["history"].append({"objectives": new_parent_objectives, "parent_pool": new_parent_pool})
    return state


def should_stop(state):
    return state["generation"] == state["max_generations"]

def should_run_diversity(state):
    # Run diversity agent every 10 generations
    return (state["generation"] % 5 == 0) and (state["generation"] > 0)

def next_step_after_eval(state):
    if should_stop(state):
        return "END"
    elif should_run_diversity(state):
        return "DiversityAgent"
    else:
        return "CondenseRepair"

def build_ea_langgraph_merged():
    workflow = StateGraph(dict)
    workflow.add_node("CondenseRepair", condense_repair_agent_node)
    workflow.add_node("ApplyStrategy", apply_strategy_node)
    workflow.add_node("EvalAndSurvivor", evaluate_and_survivor_node)
    workflow.add_node("DiversityAgent", diversity_agent_node)
    

    workflow.add_edge(START, "CondenseRepair")
    workflow.add_edge("CondenseRepair", "ApplyStrategy")
    # workflow.add_edge("PrepForDebate", "AgenticDebate")
    # workflow.add_edge("AgenticDebate", "ApplyStrategy")
    workflow.add_edge("ApplyStrategy", "EvalAndSurvivor")
    workflow.add_conditional_edges("EvalAndSurvivor", next_step_after_eval, {
        "END": END,
        "DiversityAgent": "DiversityAgent",
        "CondenseRepair": "CondenseRepair",
    })
    workflow.add_edge("DiversityAgent", "ApplyStrategy")

    return workflow.compile()
