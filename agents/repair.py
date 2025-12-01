import numpy as np
import ast
from Auto_weighter.utils.NSGA_related import get_pareto_front_indices


def create_llm_condense_repair_agent(llm, max_retries=10):
    def condense_repair_agent_node(state):
        parent_pool = state["parent_pool"]                 # (N, n_vars)
        parent_objectives = state["parent_objectives"]     # (N, n_objs)
        n_vars = parent_pool.shape[1]
        n_objs = parent_objectives.shape[1]
        bounds = state.get("bounds", None)

        if bounds is not None:
            lower, upper = bounds
            bounds_str = f"[{float(np.min(lower))}, {float(np.max(upper))}]"
        else:
            bounds_str = "[0, 1]"

        # --- Pareto front detection ---
        cur_epsilon=state.get("new_epsilon", state["eps_vals"])
        pareto_idx = get_pareto_front_indices(parent_objectives, epsilon=cur_epsilon)
        bad_idx = np.setdiff1d(np.arange(len(parent_pool)), pareto_idx)
        

        # --- Case 1: all points are Pareto → ask LLM for epsilon proposal ---
        if len(bad_idx) == 0:
            pool_size = parent_pool.shape[0]
            lower_target = max(1, int(0.2 * pool_size))
            upper_target = int(0.5 * pool_size)

            obj_mean = parent_objectives.mean(axis=0)
            obj_std  = parent_objectives.std(axis=0)
            obj_min  = parent_objectives.min(axis=0)
            obj_max  = parent_objectives.max(axis=0)

            system_message = (
                "System: You are assisting a multi-objective evolutionary algorithm.\n"
                "Currently, ALL candidate solutions are on the Pareto front.\n\n"
                "Your task:\n"
                f"- Propose a new epsilon value. Epsilon is the tolerance margin used in "
                f"Pareto dominance checks: when comparing two objective vectors, solution A "
                f"is considered to dominate B if A is better in at least one objective and "
                f"no worse than B in others by more than epsilon.\n"
                f"- Goal: reduce the Pareto front size from {pool_size} to between "
                f"{lower_target} and {upper_target}.\n"
                "- The epsilon must be a single positive float.\n\n"
                "Output format:\n"
                "- Return ONLY the float (no explanation, no list)."
            )

            user_msg = (
                "\n==== Parent Objectives (RMS errors) ====\n" 
                + arr2str(parent_objectives, decimals=3, max_rows=20)
                + "\n==== Statistics across pool (per objective) ====\n"
                f"- Mean: {obj_mean}\n"
                f"- Std:  {obj_std}\n"
                f"- Min:  {obj_min}\n"
                f"- Max:  {obj_max}\n\n"
                f"- Current epsilon: {cur_epsilon}\n"
                f"Given these values, propose ONLY a single positive epsilon "
                f"so that the Pareto front size falls between {lower_target} and {upper_target}."
            )

            tries, new_epsilon = 0, None
            while new_epsilon is None and tries < max_retries:
                result = llm([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_msg}
                ])
                try:
                    new_epsilon = float(result.strip())
                except Exception:
                    new_epsilon = None
                tries += 1

            if new_epsilon is None:
                print("⚠️ LLM failed to propose epsilon, defaulting to 0.1")
                new_epsilon = 0.1

            return {
                **state,
                "new_epsilon": new_epsilon,
                "condensed_pool": parent_pool,
                "rationales": ["No repair: all Pareto"] 
            }

        # --- Case 2: some dominated → run original repair agent ---
        else:

            # --- Step 2: Summaries ---
            param_param_corr = state["summary"]["param_param_corr"]
            param_obj_corr = state["summary"]["param_obj_corr"]
            pca_loadings = state["summary"]["pca_loadings"]
            pca_explained_variance = state["summary"]["pca_explained_variance"]
            budget = state.get("budget", 2)

            param_diversity = np.std(parent_pool, axis=0)

            # --- Step 3: Bad set summary per objective ---
            bad_summary_lines = []
            for idx in bad_idx:
                bad_dims = np.where(parent_objectives[idx] < 20)[0].tolist()
                bad_summary_lines.append(
                    f"- Row {idx}: has objectives more than threshold in those index {bad_dims}; "
                    f"params={np.array2string(parent_pool[idx], precision=3, separator=', ')}, "
                    f"objs={np.array2string(parent_objectives[idx], precision=3, separator=', ')}"
                )
            
            bad_summary = "\n".join(bad_summary_lines) if len(bad_summary_lines) else "None found."
            system_message = (
                "System: You are an optimization agent tuning parameters (σ array) for a multi-objective evolutionary algorithm.\n\n"
                "Problem summary:\n"
                f"- There are {n_vars} datasets, each with its own RMS error objective e_k (lower is better).\n"
                f"- Each candidate parameter vector has length {n_vars}: per-dataset scale parameters σ_k.\n"
                f"- Each reconstruction yields an objective vector of length {n_objs}: RMS residuals e_k for each dataset (lower is better).\n\n"
                f"- The overall goal is to minimize all RMS objectives as much as possible (multi-objective minimization) by repairing the bad parameter sets (adjust the σ values to proper values).\n"
                "How parameters drive objectives:\n"
                "- The parameters σ_k act as scaling factors in the minimization process.\n"
                "- Smaller σ_k → dataset k has more influence, which will reduce its error but risks overfitting its noise and hurting other datasets.\n"
                "- Larger σ_k → dataset k has less influence, which will prevent overfitting but can leave its error high.\n"
                "- Your job:find σ values that reduce all RMS objectives without collapsing into overfitting on one dataset or ignoring others.\n\n"
                "What you will be given, and how it can help:\n"
                "1) Full parameter pool and objectives.\n"
                "2) Summary of BAD sets (non-Pareto points with high RMS on specific objectives).\n"
                "3) Parameter–parameter correlation (how σ interact with each other).\n"
                "   • Entry [i, j] is the correlation between σ_i and σ_j across the population.\n"
                "   • Positive correlation: when σ_i is high, σ_j also tends to be high (they move together).\n"
                "   • Negative correlation: when σ_i is high, σ_j tends to be low (they trade off).\n"
                "   • Near zero: σ_i and σ_j vary independently.\n"
                "   • This reveals which σ often move together; Use them during your proposal if needed.\n"
                "4) Parameter–objective correlation (how σ affect dataset errors).\n"
                "   • Entry [i, j] is the correlation between σ_i and objective j (RMS error of dataset j).\n"
                "   • Shows how each σ influences each objective.\n"
                "   • To improve a target objective k, do not only consider σ_k itself—look at other σ_j that are strongly correlated with objective k.\n"
                "   • For example: if objective k is high and σ_j has a strong positive correlation with objective k, decreasing σ_j may also reduce objective k "
                "5) PCA loadings + explained variance (main directions of variation).\n"
                "6) Diversity scores per parameter (spread of values).\n"
                "   • The diversity score array has one value per parameter σ_k (position i → σ_i).\n"
                "   • Each score measures spread of σ values across the current pool (e.g., normalized std).\n"
                "   • Low score = little variation (values clustered) → encourage exploration: try different ranges.\n"
                "   • High score = high variation (values spread) → encourage exploitation: refine near good regions.\n"
                "   • Use these scores to decide which σ to perturb and by how much.\n"
                f"7) Bounds: all σ must remain strictly inside {bounds_str}.\n"
                f"8) Repair budget: you may adjust at most {budget} parameters per bad set.\n\n"
                "**Guidelines:**\n"
                "- You should aim for all objectives to be as low as possible (<20).\n"
                "- Use these facts to identify which parameters to change, by how much, and why\n"
                "- Learn from the correlations, PCA, and diversity to make meaningful edits (either big or small) to the parameters.\n"
                "- Focus on reducing errors for bad sets while maintaining balance.\n"
                "- Do not collapse all σ to extremes (0 or max).\n"
                "Output format (STRICT):\n"
                f"- Return a valid Python list of {len(bad_idx)} dicts.\n"
                "- the current pareto sets might be a temporary solution during the exploration and don't trust their objectives in terms of what is optimal.\n"
                f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
                "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
            )

            # --- Step 5: User Prompt ---
            user_msg = (
                "\n==== Indexing & Semantics ====\n"
                f"• Parameter indices: 0..{n_vars-1} (σ_k per dataset).\n"
                f"• Objective indices:  0..{n_objs-1} (RMS error per dataset; lower is better).\n\n"
                "==== Non-Pareto BAD Sets (to be repaired) ====\n"
                + bad_summary
                + "\n\n==== Diversity per Parameter ====\n"
                + arr2str(param_diversity)
                + "\n\n==== Global Parameter–Parameter Correlation ====\n"
                + arr2str(param_param_corr)
                + "\n\n==== Global Parameter–Objective Correlation ====\n"
                + arr2str(param_obj_corr)
                + "\n\n==== PCA Loadings + Explained Variance ====\n"
                + arr2str(pca_loadings) + "\n"
                + arr2str(pca_explained_variance)
                + "\n\nInstructions:\n"
                "• Use correlations, PCA, and diversity to guide your edits.\n"
                f"• Adjust at most {budget} parameters per bad set.\n"
                f"• Keep all σ strictly within {bounds_str}.\n\n"
                "**Output format (STRICT):**\n"
                "• Focus on reducing global RMS errors.\n"
                f"- Return a valid Python list of {len(bad_idx)} dicts.\n"
                f"- Each dict must have 'values' (a list of {n_vars} floats) and 'rationale' (short text).\n"
                "- The FIRST LINE of your reply must be ONLY that Python list—no extra text."
            )

            tries = 0
            sets = None
            report_raw = ""
            while sets is None and tries < max_retries:
                prompt = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_msg}
                ]
                result = llm(prompt)
                report_raw = result
                try:
                    sets = ast.literal_eval(result.strip())
                    valid = (
                        isinstance(sets, list)
                        and all(
                            isinstance(item, dict)
                            and 'values' in item
                            and 'rationale' in item
                            and isinstance(item['values'], (list, tuple, np.ndarray))
                            and len(item['values']) == n_vars
                            for item in sets
                        )
                        and len(sets) == len(bad_idx)   # ✅ only bad sets now
                    )
                    if not valid:
                        sets = None
                except Exception:
                    sets = None

                if sets is None:
                    system_message += (
                        f"\nWARNING: Your previous output was NOT a valid Python list of {len(bad_idx)} dicts "
                        f"with 'values' (length {n_vars}) and 'rationale'. "
                        "The first line must be ONLY that Python list. Try again."
                    )
                tries += 1
            if sets is None:
                print("⚠️ LLM repair agent failed, returning unchanged pool.")
                return {
                    **state,
                    "condensed_pool": parent_pool,
                    "rationales": ["LLM repair failed, no changes made."]
            }

            arrs = [np.array(item['values'], dtype=float) for item in sets]
            rationales = [str(item['rationale']) for item in sets]

            # ✅ Merge repaired bad sets with original Pareto front
            new_pool = np.vstack([parent_pool[pareto_idx], np.vstack(arrs)])

            return {
                **state,
                "condensed_pool": new_pool,
                "rationales": rationales,
            }
    return condense_repair_agent_node