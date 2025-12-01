import numpy as np
from sklearn.decomposition import PCA


def crowding_distance(objectives, front):
    if len(front) == 0:
        return np.array([])
    obj = objectives[front]
    n_obj = obj.shape[1]
    distance = np.zeros(len(front))
    for m in range(n_obj):
        sorted_idx = np.argsort(obj[:, m])
        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
        min_m, max_m = obj[sorted_idx[0], m], obj[sorted_idx[-1], m]
        if max_m - min_m == 0:
            continue
        for i in range(1, len(front) - 1):
            distance[sorted_idx[i]] += (
                obj[sorted_idx[i + 1], m] - obj[sorted_idx[i - 1], m]
            ) / (max_m - min_m)
    return distance

def select_parent_indices(rng, objectives, n_select):
    N = objectives.shape[0]
    fronts = nsga2_pareto_fronts(objectives)
    rank = np.zeros(N, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    crowd_dist = np.zeros(N)
    for front in fronts:
        cd = crowding_distance(objectives, front)
        crowd_dist[front] = cd

    selected_indices = []
    for _ in range(n_select):
        idx1, idx2 = rng.choice(N, 2, replace=False)
        if rank[idx1] < rank[idx2]:
            selected = idx1
        elif rank[idx2] < rank[idx1]:
            selected = idx2
        else:
            if crowd_dist[idx1] > crowd_dist[idx2]:
                selected = idx1
            elif crowd_dist[idx2] > crowd_dist[idx1]:
                selected = idx2
            else:
                selected = rng.choice([idx1, idx2])
        selected_indices.append(selected)
    return selected_indices



def initialize_gaussian_pool(rng,center_point, n_samples, std, bounds=None):
    """
    center_point: 1D np.ndarray, shape (D,)
    n_samples: int, number of samples to generate
    std: float or array, standard deviation for Gaussian sampling
    bounds: tuple (lower, upper), both np.ndarray of shape (D,); optional, to clip samples within bounds
    Returns: parameter_pool (n_samples, D)
    """
    D = center_point.shape[0]
    sigma = std * np.abs(center_point)
    centered_samples = rng.normal(center_point, sigma, size=(n_samples, D))
    if bounds is not None:
        lower, upper = bounds
        pool = np.clip(centered_samples, lower, upper)
    return pool

def summarize_population(all_param_pool, all_objectives, n_pca_components=5):

    N, D_param = all_param_pool.shape
    D_obj = all_objectives.shape[1]

    # 1. Parameter–Parameter correlation matrix
    param_param_corr = np.corrcoef(all_param_pool, rowvar=False)  # (D_param, D_param)

    # 2. Parameter–Objective correlation matrix
    param_obj_corr = np.zeros((D_param, D_obj))
    for i in range(D_param):
        for j in range(D_obj):
            p = all_param_pool[:, i]
            o = all_objectives[:, j]
            mask = np.isfinite(p) & np.isfinite(o)
            if np.sum(mask) > 2:
                param_obj_corr[i, j] = np.corrcoef(p[mask], o[mask])[0, 1]
            else:
                param_obj_corr[i, j] = np.nan

    # 3. PCA on parameter pool
    n_pca = min(n_pca_components, D_param, N)
    pca = PCA(n_components=n_pca)
    pca.fit(all_param_pool)
    pca_loadings = pca.components_   # (n_pca, D_param)
    explained_variance = pca.explained_variance_ratio_  # (n_pca,)

    # 4. Summary statistics (mean, std, min, max)
    # param_mean = np.mean(all_param_pool, axis=0)
    # param_std = np.std(all_param_pool, axis=0)
    # param_min = np.min(all_param_pool, axis=0)
    # param_max = np.max(all_param_pool, axis=0)
    # obj_mean = np.mean(all_objectives, axis=0)
    # obj_std = np.std(all_objectives, axis=0)
    # obj_min = np.min(all_objectives, axis=0)
    # obj_max = np.max(all_objectives, axis=0)

    summary = {
        "param_param_corr": param_param_corr,
        "param_obj_corr": param_obj_corr,
        "pca_loadings": pca_loadings,
        "pca_explained_variance": explained_variance,
    }
    return summary

def get_pareto_front_indices(objectives, epsilon=0.1, max_allowed=None):
    # 1) Optional filtering
    objs = objectives
    if max_allowed is not None:
        mask = (objs <= np.broadcast_to(max_allowed, objs.shape[1])).all(axis=1)
        idxs = np.nonzero(mask)[0]
        objs = objs[mask]
    else:
        idxs = np.arange(len(objs))

    # 2) Find dominated points
    dominated = np.zeros(len(objs), dtype=bool)
    for i, oi in enumerate(objs):
        # vectorized check: which j strictly dominates i?
        leq = objs <= oi + epsilon
        lt  = objs <  oi - epsilon
        dom = np.all(leq, axis=1) & np.any(lt, axis=1)
        dom[i] = False  # ignore self
        if dom.any():
            dominated[i] = True

    # 3) Return original indices of the non‑dominated ones
    return idxs[~dominated]



def get_bounds_and_constraints(bounds, constraints=None):
    lower, upper = bounds
    s = f"After the edits, all values in the final parameter pool should be bounded in the range [{lower[0]}, {upper[0]}].\n"
    if constraints:
        if isinstance(constraints, list):
            constraints_str = "; ".join(constraints)
        else:
            constraints_str = str(constraints)
        s += f"Additional constraints: {constraints_str}\n"
    return s

def get_index_mapping_note(n_params):
    return (
        f"Parameter index N (from 0 to {n_params-1}) always refers to the N-th parameter "
        "in every array, group, and summary. Objective indices are likewise aligned. "
        "Use this convention for all operator or strategy selection."
    )


def arr2str(arr, decimals=3, max_rows=50):
            arr = np.array(arr)
            n_rows = arr.shape[0] if arr.ndim > 1 else 1
            if n_rows > max_rows:
                arr = arr[:max_rows]
                note = f" [showing first {max_rows} rows]\n"
            else:
                note = ""
            return np.array2string(arr, max_line_width=120, precision=decimals, separator=", ") + note


def nsga2_pareto_fronts(objectives, epsilon=50, max_allowed=None):
    """
    Returns a list of fronts (each front is a list of indices).
    If epsilon > 0, uses epsilon-dominance.
    If max_allowed is set (float or array), filters out points exceeding this in any objective.
    """

    N = objectives.shape[0]
    valid = np.ones(N, dtype=bool)
    if max_allowed is not None:
        # Exclude points with any objective > max_allowed (can be scalar or array-like)
        max_allowed = np.broadcast_to(max_allowed, objectives.shape[1])
        valid = (objectives <= max_allowed).all(axis=1)
    indices = np.arange(N)[valid]
    objs = objectives[indices]

    domination_count = np.zeros(len(indices), dtype=int)
    dominated = [[] for _ in range(len(indices))]
    fronts = [[]]

    for p in range(len(indices)):
        for q in range(len(indices)):
            if p == q:
                continue
            if epsilon > 0.0:
                # Epsilon-dominance
                if (np.all(objs[p] <= objs[q] + epsilon) and
                    np.any(objs[p] < objs[q] - epsilon)):
                    dominated[p].append(q)
                elif (np.all(objs[q] <= objs[p] + epsilon) and
                      np.any(objs[q] < objs[p] - epsilon)):
                    domination_count[p] += 1
            else:
                if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                    dominated[p].append(q)
                elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                    domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(int(indices[p]))  # Use original indices

    i = 0
    while fronts[i]:
        next_front = []
        for idx in fronts[i]:
            p = np.where(indices == idx)[0][0]  # local index in objs
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(int(indices[q]))  # Use original indices
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def nsga2_crowding_distance(objectives, front):
    if len(front) == 0:
        return np.array([])
    obj = objectives[front]
    n_obj = obj.shape[1]
    distance = np.zeros(len(front))
    for m in range(n_obj):
        sorted_idx = np.argsort(obj[:, m])
        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
        min_m, max_m = obj[sorted_idx[0], m], obj[sorted_idx[-1], m]
        if max_m - min_m == 0:
            continue
        for i in range(1, len(front) - 1):
            distance[sorted_idx[i]] += (
                obj[sorted_idx[i + 1], m] - obj[sorted_idx[i - 1], m]
            ) / (max_m - min_m)
    return distance


def rank_based_parent_selection(rng, objectives, n_parents):
    """
    Returns n_parents indices, sampling more from top fronts and within them by crowding distance.
    """
    fronts = nsga2_pareto_fronts(objectives)
    # Assign a probability for each front (front 0 gets highest)
    front_weights = np.linspace(1.0, 0.2, num=max(2, len(fronts)))  # Decay front probabilities
    prob = np.zeros(objectives.shape[0])
    for i, front in enumerate(fronts):
        dists = nsga2_crowding_distance(objectives, front)
        dists = np.nan_to_num(dists, nan=0.0, posinf=1.0)  # treat inf as 1.0 (most preferred)
        dists += 1e-6  # ensure positive
        p = front_weights[i] * dists / (np.sum(dists) if np.sum(dists) > 0 else 1)
        prob[front] = p
    prob = prob / prob.sum()
    return rng.choice(objectives.shape[0], size=n_parents, replace=True, p=prob)


def sbx_crossover(
    rng,
    parent_pool,
    parent_pairs,
    eta=15,
    bounds=None,
    prob_var=1.0,
    prob_bin=0.5,
    eps=1.0e-14,
    return_both=False,
):
    # --- setup & shapes ---
    parent_pairs = np.asarray(parent_pairs)
    M = parent_pairs.shape[0]
    D = parent_pool.shape[1]

    if bounds is None:
        lower = np.zeros(D, dtype=float)
        upper = np.ones(D, dtype=float)
    else:
        lower, upper = (np.asarray(bounds[0], dtype=float),
                        np.asarray(bounds[1], dtype=float))

    # broadcast helpers
    def _to_vec(x):
        x = np.asarray(x)
        if x.ndim == 0:
            x = np.full(D, x, dtype=float)
        return x.astype(float, copy=False)

    eta_vec = _to_vec(eta)
    prob_var_vec = _to_vec(prob_var)
    prob_bin_vec = _to_vec(prob_bin)

    # gather parents -> X shape (2, M, D)
    pidx1 = parent_pairs[:, 0]
    pidx2 = parent_pairs[:, 1]
    X = np.stack([parent_pool[pidx1], parent_pool[pidx2]], axis=0)

    # per-variable crossover activation for each mating
    cross = rng.random((M, D)) < prob_var_vec  # (M, D)

    # skip variables where parents are "too close"
    too_close = np.abs(X[0] - X[1]) <= eps       # (M, D)
    cross[too_close] = False

    # skip variables whose lower==upper
    same_bound = (lower == upper)                # (D,)
    if np.any(same_bound):
        cross[:, same_bound] = False

    # prepare y1<=y2 ordering for SBX math, plus bounds & parameters, only where cross==True
    p1 = X[0][cross]
    p2 = X[1][cross]
    sm = p1 < p2
    y1 = np.where(sm, p1, p2)
    y2 = np.where(sm, p2, p1)

    # bounds/params repeated for all matings, then pick where cross
    _lower = np.repeat(lower[None, :], M, axis=0)[cross]
    _upper = np.repeat(upper[None, :], M, axis=0)[cross]
    _eta   = np.repeat(eta_vec[None, :], M, axis=0)[cross]
    _pbin  = np.repeat(prob_bin_vec[None, :], M, axis=0)[cross]

    # randoms per crossed gene
    rand = rng.random(len(_eta))

    def calc_betaq(beta):
        alpha = 2.0 - np.power(beta, -(_eta + 1.0))
        mask = rand <= (1.0 / alpha)
        betaq = np.empty_like(alpha)
        # branch 1
        betaq[mask] = np.power(rand[mask] * alpha[mask], 1.0 / (_eta[mask] + 1.0))
        # branch 2
        betaq[~mask] = np.power(1.0 / (2.0 - rand[~mask] * alpha[~mask]),
                                1.0 / (_eta[~mask] + 1.0))
        return betaq

    # SBX math
    delta = (y2 - y1)  # difference
    # guard: any residual zero delta (shouldn't happen after too_close mask)
    nz = np.abs(delta) > eps
    if not np.all(nz):
        # where delta==0, just copy parent values
        # create placeholders to fill and then overwrite where nz
        c1_all = y1.copy()
        c2_all = y2.copy()
        if np.any(nz):
            # left side
            beta = 1.0 + (2.0 * (y1[nz] - _lower[nz]) / delta[nz])
            betaq = calc_betaq(beta)
            c1 = 0.5 * ((y1[nz] + y2[nz]) - betaq * delta[nz])
            # right side
            beta = 1.0 + (2.0 * (_upper[nz] - y2[nz]) / delta[nz])
            betaq = calc_betaq(beta)
            c2 = 0.5 * ((y1[nz] + y2[nz]) + betaq * delta[nz])
            c1_all[nz] = c1
            c2_all[nz] = c2
    else:
        beta = 1.0 + (2.0 * (y1 - _lower) / delta)
        betaq = calc_betaq(beta)
        c1_all = 0.5 * ((y1 + y2) - betaq * delta)

        beta = 1.0 + (2.0 * (_upper - y2) / delta)
        betaq = calc_betaq(beta)
        c2_all = 0.5 * ((y1 + y2) + betaq * delta)

    # assign children back to parent order (child1 aligns with parent X[0], child2 with X[1])
    child1_cross = np.where(sm, c1_all, c2_all)
    child2_cross = np.where(sm, c2_all, c1_all)

    # exchange children with probability prob_bin (and keep ordering stable w.r.t parent values)
    swap_mask = np.bitwise_xor(
        rng.random(len(_pbin)) < _pbin,
        X[0, cross] > X[1, cross]
    )
    child1_cross, child2_cross = np.where(swap_mask, child2_cross, child1_cross), \
                                 np.where(swap_mask, child1_cross, child2_cross)

    # start with copies of the parents
    child1 = X[0].copy()
    child2 = X[1].copy()

    # fill crossed positions
    child1[cross] = child1_cross
    child2[cross] = child2_cross

    # clamp to bounds
    np.clip(child1, lower, upper, out=child1)
    np.clip(child2, lower, upper, out=child2)

    if return_both:
        # return both children per pair
        return np.vstack([child1, child2])
    else:
        # return a single child per pair (choose child1 for determinism/backward-compat)
        return child1


def gaussian_mutation(rng, offsprings, std=0.05, bounds=None):
    mutated = offsprings + rng.normal(0, std, size=offsprings.shape)
    if bounds is not None:
        lower, upper = bounds
        mutated = np.clip(mutated, lower, upper)
    return mutated



# 4. Full strategy executor
def apply_strategy_rank_based(rng,parent_pool, parent_objectives, bounds, mutation_std=0.05):

    offsprings = []
    n_sbx = 10
    idxs1 = rank_based_parent_selection(rng, parent_objectives, n_sbx)
    idxs2 = rank_based_parent_selection(rng, parent_objectives, n_sbx)
    parent_pairs = list(zip(idxs1, idxs2))
    children = sbx_crossover(rng, parent_pool, parent_pairs, bounds=bounds)
    children = gaussian_mutation(rng, children, std=mutation_std, bounds=bounds)
    offsprings.append(children)

    if offsprings:
        result = np.vstack(offsprings)
    else:
        result = np.empty((0, parent_pool.shape[1]))
    return result#, op_labels

def nsga2_survivor_selection(parent_pool, offspring_pool, parent_objectives, offspring_objectives, pool_size):
    """
    parent_pool: (N, D), offspring_pool: (N, D)
    parent_objectives: (N, M), offspring_objectives: (N, M)
    pool_size: int (e.g. 20)
    Returns:
        new_parent_pool: (pool_size, D)
        new_parent_objectives: (pool_size, M)
    """
    # 1. Combine
    all_params = np.vstack([parent_pool, offspring_pool])       # (2N, D)
    all_objs = np.vstack([parent_objectives, offspring_objectives])  # (2N, M)
    
    # 2. Get fronts
    fronts = nsga2_pareto_fronts(all_objs)
    
    survivors = []
    for front in fronts:
        if len(survivors) + len(front) <= pool_size:
            survivors.extend(front)
        else:
            # Need to select only part of this front
            remaining = pool_size - len(survivors)
            distances = nsga2_crowding_distance(all_objs, front)
            # Select indices of the front with highest crowding distance
            idx_sorted = np.argsort(-distances)  # Descending
            chosen = [front[i] for i in idx_sorted[:remaining]]
            survivors.extend(chosen)
            break  # Pool filled

    survivors = np.array(survivors)
    new_parent_pool = all_params[survivors]
    new_parent_objectives = all_objs[survivors]
    return new_parent_pool, new_parent_objectives

# -------- helpers --------

def _randomized_argsort_desc(values, rng):
    """Sort by value DESC with random tie-breaker."""
    values = np.asarray(values, dtype=float)
    keys0 = -values
    keys1 = rng.random(values.shape[0])
    # lexsort uses last key as primary; first by keys0, then keys1
    idx = np.lexsort((keys1, keys0))
    return idx  # descending by values due to minus sign


def nsga2_crowding_distance(objectives, front):
    """Crowding distance for indices in `front` (returns distances aligned with `front`)."""
    if len(front) == 0:
        return np.array([], dtype=float)
    F = objectives[front]
    n, m = F.shape
    dist = np.zeros(n, dtype=float)
    if n == 1:
        dist[0] = np.inf
        return dist

    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = F[order[0], k], F[order[-1], k]
        denom = fmax - fmin
        if denom <= 0:
            # all equal in this objective; everyone already has whatever dist from other dims
            continue
        # interior points
        for i in range(1, n - 1):
            if np.isinf(dist[order[i]]):  # boundary already ∞
                continue
            dist[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / denom
    return dist


def _dominates(p, q, eps=0.0):
    """Return True if p ε-dominates q (minimization)."""
    if eps > 0.0:
        return (np.all(p <= q + eps)) and (np.any(p - eps< q))
    else:
        return (np.all(p <= q)) and (np.any(p < q))


def nsga2_fronts(objectives, eps=0.0, n_stop_if_ranked=None, mask=None):
    """
    Fast-enough O(N^2) non-dominated sorting with optional ε-dominance
    and early-stop when `n_stop_if_ranked` individuals have been ranked.
    Returns list of fronts (each front is array of global indices).
    """
    N = objectives.shape[0]
    if mask is None:
        idx = np.arange(N)
        F = objectives
    else:
        idx = np.where(mask)[0]
        F = objectives[idx]

    M = len(idx)
    S = [[] for _ in range(M)]
    n_dom = np.zeros(M, dtype=int)
    fronts_local = [[]]

    for p in range(M):
        Sp = S[p]
        for q in range(M):
            if p == q:
                continue
            if _dominates(F[p], F[q], eps):
                Sp.append(q)
            elif _dominates(F[q], F[p], eps):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts_local[0].append(p)

    ranked = len(fronts_local[0])
    i = 0
    while fronts_local[i]:
        next_front = []
        for p in fronts_local[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts_local.append(next_front)
        ranked += len(next_front)
        if n_stop_if_ranked is not None and ranked >= n_stop_if_ranked:
            break

    # map back to global indices
    fronts = [idx[np.array(f, dtype=int)] for f in fronts_local if len(f) > 0]
    return fronts


# -------- survival (rank + crowding) --------

def nsga2_survival(rng, objectives, n_survive, epsilon=0.0, max_allowed=None):
    """
    Return survivor indices using NSGA-II rank-and-crowding (minimization).
    - ε-dominance optional
    - If max_allowed provided (scalar or (m,)), points exceeding are filtered out.
    """
    N, M = objectives.shape

    # feasibility mask (simple threshold filter if given)
    if max_allowed is not None:
        thr = np.broadcast_to(np.asarray(max_allowed, dtype=float), (M,))
        feas_mask = (objectives <= thr).all(axis=1)
    else:
        feas_mask = np.ones(N, dtype=bool)

    # fronts with early-stop once we have >= n_survive ranked
    fronts = nsga2_fronts(objectives, eps=epsilon, n_stop_if_ranked=n_survive, mask=feas_mask)

    survivors = []
    ranks = np.full(N, fill_value=np.iinfo(np.int32).max, dtype=int)
    crowd = np.full(N, fill_value=np.nan, dtype=float)

    for k, front in enumerate(fronts):
        if len(survivors) + len(front) <= n_survive:
            # take all
            d = nsga2_crowding_distance(objectives, front)
            crowd[front] = d
            ranks[front] = k
            survivors.extend(front.tolist())
        else:
            # need to pick the best by crowding distance (descending), with randomized tie-breaking
            d = nsga2_crowding_distance(objectives, front)
            keep = n_survive - len(survivors)
            order = _randomized_argsort_desc(d, rng)[:keep]
            chosen = front[order]
            crowd[front] = d
            ranks[front] = k
            survivors.extend(chosen.tolist())
            break

    return np.array(survivors, dtype=int), ranks, crowd


# -------- parent selection (binary tournament) --------

def nsga2_tournament_selection(rng, ranks, crowding, n_parents):
    """
    Binary tournament selection using (rank → lower is better, then crowding → larger is better).
    Returns indices for the mating pool (with replacement).
    """
    N = len(ranks)
    parents = np.empty(n_parents, dtype=int)
    for t in range(n_parents):
        i, j = rng.integers(0, N, size=2)
        ri, rj = ranks[i], ranks[j]
        if ri < rj:
            win = i
        elif rj < ri:
            win = j
        else:
            ci, cj = crowding[i], crowding[j]
            if ci > cj:
                win = i
            elif cj > ci:
                win = j
            else:
                win = i if rng.random() < 0.5 else j
        parents[t] = win
    return parents
