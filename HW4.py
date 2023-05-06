import random
import pulp
import numpy as np
import matplotlib.pyplot as plt


# Define the possible values of n and m
n_values = [10, 20, 40, 60, 80, 100]
m_values = [50, 60, 70, 80, 90, 100]
# n_values = [40]
# m_values = [100]

# Define the number of instances to generate
N = 10

# Define a function to generate a random instance given n and m
def generate_instance(n, m):
    clauses = []
    for j in range(m):
        K = random.randint(1, 3)
        clause = []
        for k in range(K):
            i = random.randint(1, n)
            literal = (i, random.choice([True, False]))
            clause.append(literal)
        clauses.append(clause)
    return clauses

def rahuls_algorithm(n):
    x = [0] * n  # initialize all literals to 0
    for i in range(n):
        x[i] = random.choice([0, 1])  # set x_i to 0 or 1 with equal probability
    return x

def boosted_jinals_algorithm(clauses, n):
    x = [0] * n  # initialize all literals to 0
    for i in range(n):
        a_i = sum(1 for c in clauses if (i+1, True) in c)
        b_i = sum(1 for c in clauses if (i+1, False) in c)
        if a_i + b_i >= 1 and a_i >= b_i:
            x[i] = 1
        elif a_i + b_i >= 1 and a_i < b_i:
            x[i] = 0
        else:
            x[i] = random.choice([0, 1])  # set x_i to 0 or 1 with equal probability
    return x

def compute_performance(clauses, x):
    # Compute the number of satisfied clauses given a list of literals x
    num_satisfied = 0
    for c in clauses:
        clause_satisfied = False
        for (i, value) in c:
            if (i > 0 and x[i-1] == value) or (i < 0 and x[-i-1] != value):
                clause_satisfied = True
                break
        if clause_satisfied:
            num_satisfied += 1
    return num_satisfied

def r_bj_a(clauses, n):
    x_ra = rahuls_algorithm(n)
    x_bja = boosted_jinals_algorithm(clauses, n)
    perf_ra = compute_performance(clauses, x_ra)
    perf_bja = compute_performance(clauses, x_bja)
    if perf_ra >= perf_bja:
        return x_ra
    else:
        return x_bja


def lp_based_rounding(clauses, n):
    x = [0] * n  # initialize all literals to 0
    for i in range(n):
        # Step 1: Construct the Integral Program (IP) for variable i
        prob = pulp.LpProblem(f"LR for x_{i+1}", pulp.LpMaximize)
        z = [pulp.LpVariable(f"z_{j+1}", cat="Binary") for j in range(len(clauses))]
        prob += pulp.lpSum(z)
        for j, c in enumerate(clauses):
            p_j = [v for v in c if v[0] == i+1 and v[1] is True]
            q_j = [v for v in c if v[0] == i+1 and v[1] is False]
            prob += pulp.lpSum(x[v[0]-1] for v in p_j) + pulp.lpSum(1 - x[v[0]-1] for v in q_j) >= z[j]
        # Step 2: Relax x_i and z_j to [0, 1]
        for j in range(len(clauses)):
            z[j].lowBound = 0
            z[j].upBound = 1
        # Step 3: Solve the relaxed Linear Programming (LP)
        prob.solve()
        x_i_star = prob.variables()[0].varValue
        # Step 4: Round x_i to 0 or 1 with probability x_i_star
        x[i] = 1 if random.random() < x_i_star else 0
    return x


def boosted_lp_based_rounding(clauses, n):
    x = [0] * n  # initialize all literals to 0
    for i in range(n):
        # Step 1: Construct the Integral Program (IP) for variable i
        prob = pulp.LpProblem(f"BLR for x_{i+1}", pulp.LpMaximize)
        z = [pulp.LpVariable(f"z_{j+1}", cat="Binary") for j in range(len(clauses))]
        prob += pulp.lpSum(z)
        for j, c in enumerate(clauses):
            p_j = [v for v in c if v[0] == i+1 and v[1] is True]
            q_j = [v for v in c if v[0] == i+1 and v[1] is False]
            prob += pulp.lpSum(x[v[0]-1] for v in p_j) + pulp.lpSum(1 - x[v[0]-1] for v in q_j) >= z[j]
        # Step 2: Relax x_i and z_j to [0, 1]
        for j in range(len(clauses)):
            z[j].lowBound = 0
            z[j].upBound = 1
        # Step 3: Solve the relaxed Linear Programming (LP)
        prob.solve()
        x_i_star = prob.variables()[0].varValue
        # Step 4: Round x_i to 0 or 1 with probability g(x_i_star)
        prob_g = 4**(x_i_star - 1)
        x[i] = 1 if random.random() < prob_g else 0
    return x


def solve_ip(instance, n):
    # Initialize the IP
    prob = pulp.LpProblem("Maximize_Satisfied_Clauses", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i + 1}", cat="Binary") for i in range(n)]
    z = [pulp.LpVariable(f"z_{j + 1}", cat="Binary") for j in range(len(instance))]

    # Set the objective function
    prob += pulp.lpSum(z)

    # Add the constraints
    for j, c in enumerate(instance):
        P_j = [i for i, value in c if value is True]
        Q_j = [i for i, value in c if value is False]
        prob += pulp.lpSum(x[i - 1] for i in P_j) + pulp.lpSum(1 - x[i - 1] for i in Q_j) >= z[j]

    # Solve the IP
    prob.solve()

    # Get the optimal value
    opt = pulp.value(prob.objective)

    return opt


# (The functions for the algorithms and the IP solver are not repeated here)

def run_experiments(alg_names):
    M = 20  # Number of iterations for each algorithm

    alg_funcs = [rahuls_algorithm, boosted_jinals_algorithm, r_bj_a, lp_based_rounding, boosted_lp_based_rounding]

    tau = np.zeros((len(alg_funcs), len(n_values), len(m_values)))

    for n_idx, n in enumerate(n_values):
        for m_idx, m in enumerate(m_values):
            opt_total = 0
            alg_total = np.zeros(len(alg_funcs))

            for _ in range(N):
                instance = generate_instance(n, m)
                opt = solve_ip(instance, n)
                opt_total += opt

                for alg_idx, alg_func in enumerate(alg_funcs):
                    alg_perf_total = 0

                    for _ in range(M):
                        if alg_func == rahuls_algorithm:
                            solution = alg_func(n)
                        else:
                            solution = alg_func(instance, n)

                        alg_perf = compute_performance(instance, solution)
                        alg_perf_total += alg_perf

                    alg_total[alg_idx] += alg_perf_total / M

            opt_avg = opt_total / N
            for alg_idx in range(len(alg_funcs)):
                tau[alg_idx, n_idx, m_idx] = alg_total[alg_idx] / opt_avg

    return tau

def plot_champion_algorithms(tau):
    champion_indices = np.argmax(tau, axis=0)

    fig, ax = plt.subplots()
    for n_idx, n in enumerate(n_values):
        for m_idx, m in enumerate(m_values):
            champion_idx = champion_indices[n_idx, m_idx]
            ax.scatter(n, m, marker="s", s=100)
            ax.text(n, m, str(champion_idx + 1), ha="center", va="center", fontsize=12)

    ax.set_xlabel("n")
    ax.set_ylabel("m")
    ax.set_title("Champion Algorithms")

    plt.show()

alg_names = ["RA", "BJA", "R-BJ-A", "LR", "B-LR"]
tau = run_experiments(alg_names)
plot_champion_algorithms(tau)

# Print the tables of τ_l(n, m) for each algorithm
for l, name in enumerate(alg_names):
    print(f"τ_{name}(n, m):")
    print(tau[l])




# Test rahuls_algorithm
# x = rahuls_algorithm(10)
# print(x)

# # Test boosted_jinals_algorithm
# clauses = [
#     [(1, True), (3, False), (4, True)],
#     [(2, False), (3, True), (5, True)],
#     [(1, False), (2, True), (3, False), (5, False)],
#     [(1, True), (3, True), (4, False)],
#     [(2, False), (4, True), (5, False)],
#     [(1, False), (2, True), (4, False)],
#     [(2, True), (3, False), (5, True)],
#     [(1, True), (2, False), (3, True), (5, True)],
#     [(1, False), (4, True), (5, False)],
#     [(2, True), (4, False), (5, True)],
# ]
# x = boosted_jinals_algorithm(clauses, 10)
# print(x)

# # Test r_bj_a
# clauses = [
#     [(1, True), (3, False), (4, True)],
#     [(2, False), (3, True), (5, True)],
#     [(1, False), (2, True), (3, False), (5, False)],
#     [(1, True), (3, True), (4, False)],
#     [(2, False), (4, True), (5, False)],
#     [(1, False), (2, True), (4, False)],
#     [(2, True), (3, False), (5, True)],
#     [(1, True), (2, False), (3, True), (5, True)],
#     [(1, False), (4, True), (5, False)],
#     [(2, True), (4, False), (5, True)],
# ]
# x = r_bj_a(clauses, 10)
# print(x)

# # Test lp_based_rounding
# clauses = [
#     [(1, True), (3, False), (4, True)],
#     [(2, False), (3, True), (5, True)],
#     [(1, False), (2, True), (3, False), (5, False)],
#     [(1, True), (3, True), (4, False)],
#     [(2, False), (4, True), (5, False)],
#     [(1, False), (2, True), (4, False)],
#     [(2, True), (3, False), (5, True)],
#     [(1, True), (2, False), (3, True), (5, True)],
#     [(1, False), (4, True), (5, False)],
#     [(2, True), (4, False), (5, True)],
# ]
# x = lp_based_rounding(clauses, 10)
# print(x)

# # Test boosted_lp_based_rounding
# clauses = [
#     [(1, True), (3, False), (4, True)],
#     [(2, False), (3, True), (5, True)],
#     [(1, False), (2, True), (3, False), (5, False)],
#     [(1, True), (3, True), (4, False)],
#     [(2, False), (4, True), (5, False)],
#     [(1, False), (2, True), (4, False)],
#     [(2, True), (3, False), (5, True)],
#     [(1, True), (2, False), (3, True), (5, True)],
#     [(1, False), (4, True), (5, False)],
#     [(2, True), (4, False), (5, True)],
# ]
# x = boosted_lp_based_rounding(clauses, 10)
# print(x)

