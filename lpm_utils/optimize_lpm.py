import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def optimize_lpm(z, num_feature, num_obs, x_order, num_order, v_order, max_point, max_features, C0, relax=False, model_display=False):
    
    model = gp.Model()
    model.Params.TimeLimit=7200
    model.Params.MIPGap=1e-2
    model.params.NonConvex=2
    model.params.presolve=0

    # ============================================================================================
    # Variables


    theta = {}
    q = {}
    v = {}
    u0 = model.addVar(lb=-max_point, ub=0, vtype=GRB.INTEGER, name="u0")
    u = {}
    u_tilde = {}
    eta = {}
    phi = {}

    for j in range(num_feature):
        for t in range(num_order[j]):
            for p in range(max_point + 1):
                theta[(j, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"theta_{j}_{t}_{p}")

    for j in range(num_feature):
        for p in range(max_point + 1):
            q[(j, p)] = model.addVar(vtype=GRB.BINARY, name=f"q_{j}_{p}")
        v[j] = model.addVar(vtype=GRB.BINARY, name=f"v_{j}")
        
    for i in range(num_obs):
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")

        u_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")


    # ============================================================================================
    # Constraints


    cons1 = {}
    for j in range(num_feature):
        for t in range(num_order[j] - 1):
            for p in range(max_point + 1):
                cons1[(j, t, p)] = model.addConstr(theta[(j, t, p)] <= theta[(j, t + 1, p)])

    cons2 = {}
    cons2_0 = {}
    for j in range(num_feature):
        for p in range(max_point + 1):
            cons2[(j, p)] = model.addConstr(theta[(j, num_order[j] - 1, p)] == q[(j, p)])
            cons2_0[(j, p)] = model.addConstr(theta[(j, 0, p)] == 0)

    cons3 = {}
    for j in range(num_feature):
        cons3[j] = model.addConstr(gp.quicksum(q[(j, p)] for p in range(max_point + 1)) == v[j])

    model.addConstr(gp.quicksum(v[j] for j in range(num_feature)) <= max_features)

    cons4 = {}
    for i in range(num_obs):
        cons4[i] = model.addConstr(u[i] == u0 + gp.quicksum(
            p * gp.quicksum(theta[(j, v_order[j][i], p)] for j in range(num_feature))
            for p in range(max_point + 1)
            ))

    cons5 = {}
    for i in range(num_obs):
        cons5[i] = model.addConstr(u_tilde[i] == u[i] - u0)


    # ============================================================================================
    # Objective

    gamma = max_point * max_features

    # obj = (
    #     (gp.quicksum(u[i]**2 / gamma**2 - 2 * z[i] * u[i] / gamma for i in range(num_obs)) / num_obs)
    #     + C0 * gp.quicksum(v[j] for j in range(num_feature))
    # )

    obj = (
        (gp.quicksum(u_tilde[i]**2 / gamma**2 - 2 * z[i] * u_tilde[i] / gamma for i in range(num_obs)) / num_obs)
        + C0 * gp.quicksum(v[j] for j in range(num_feature))
    )

    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.update()

    if relax:
        for var in model.getVars():
            if var.vType == GRB.BINARY:
                var.vType = GRB.CONTINUOUS
                var.setAttr(GRB.Attr.LB, 0.0)
                var.setAttr(GRB.Attr.UB, 1.0)

    model.update()
    model.optimize()

    # ============================================================================================


    if model.status == GRB.OPTIMAL:
        print("Solution is optimal")
    elif model.status == GRB.TIME_LIMIT:
        print("Solution is suboptimal due to a time limit, but a primal solution is available")
    else:
        print("Something else")

    print("Objective value:", model.objVal)

    u0_solution = u0.x
    theta_solution = {(j, t, p): theta[j, t, p].x for (j, t, p) in theta}
    
    if model_display: print(model.display())

    return u0_solution, theta_solution, gamma


