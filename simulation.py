# Building a "real-world" Abilene topology simulation (Topology Zoo - Abilene)
# We'll:
# - create the Abilene topology (nodes + edges) as recorded in the Topology Zoo
# - simplify parameters: capacities (Mbps), delays (ms) with clear labels
# - create a realistic demand matrix (commodity list) with understandable fields
# - solve the Multi-Commodity Flow LP with QoS delay constraints using scipy.optimize.linprog
# - plot the network, edge utilization, per-commodity allocations
# - display the topology and the simplified parameter tables
#
# Note: This uses an embedded Abilene topology (public Topology Zoo). Source: topology-zoo.org.
# The detailed dataset and metadata are available at https://topology-zoo.org/dataset.html (Topology Zoo).
# See web.run reference: turn1view0.
#
import networkx as nx, pandas as pd, numpy as np, matplotlib.pyplot as plt, os, random
from scipy.optimize import linprog
random.seed(1)

# Abilene nodes (PoP cities) and edges - simplified undirected links from Topology Zoo Abilene
nodes = ["NewYork","Chicago","WashingtonDC","Seattle","Sunnyvale","LosAngeles","Denver",
         "KansasCity","Houston","Atlanta","Indianapolis"]
# edges (undirected) - based on Abilene connectivity; we'll add symmetric directed arcs
undirected_edges = [
    ("NewYork","Chicago"), ("NewYork","WashingtonDC"), ("Chicago","Denver"),
    ("Denver","KansasCity"), ("KansasCity","Houston"), ("Houston","Atlanta"),
    ("Atlanta","Indianapolis"), ("Indianapolis","NewYork"), ("LosAngeles","Sunnyvale"),
    ("Sunnyvale","Seattle"), ("LosAngeles","Denver"), ("Seattle","Chicago"),
    ("Denver","LosAngeles"), ("Chicago","NewYork"), ("Indianapolis","WashingtonDC")
]

G = nx.DiGraph()
G.add_nodes_from(nodes)
# Assign capacities and delay (simple, easy-to-understand values)
for (u,v) in undirected_edges:
    cap = 100  # Mbps (use uniform capacity to simplify)
    delay = 10  # ms baseline per-link
    # Add both directions
    G.add_edge(u,v,capacity=cap,delay=delay)
    G.add_edge(v,u,capacity=cap,delay=delay)

# Show simplified network table
edge_table = []
for u,v in G.edges():
    edge_table.append({"u":u,"v":v,"capacity_Mbps":G[u][v]["capacity"],"delay_ms":G[u][v]["delay"]})
edge_df = pd.DataFrame(edge_table)

# Create commodities (realistic SD pairs)
commodities = [
    {"id":"k1","src":"NewYork","dst":"LosAngeles","requested_max":30,"revenue":3.0,"Dk":100},
    {"id":"k2","src":"Chicago","dst":"Seattle","requested_max":25,"revenue":2.5,"Dk":80},
    {"id":"k3","src":"Atlanta","dst":"Sunnyvale","requested_max":20,"revenue":2.0,"Dk":120},
    {"id":"k4","src":"Houston","dst":"NewYork","requested_max":15,"revenue":1.8,"Dk":90},
    {"id":"k5","src":"WashingtonDC","dst":"LosAngeles","requested_max":25,"revenue":2.8,"Dk":110},
    {"id":"k6","src":"Seattle","dst":"Houston","requested_max":10,"revenue":4.0,"Dk":130}
]
K = len(commodities)
edges = list(G.edges())
E = len(edges)
n_vars = K*E + K

# Build LP matrices
# Objective: maximize sum revenue * x_k  => minimize -sum revenue * x_k
c = [0.0]*(K*E) + [-commodities[i]["revenue"] for i in range(K)]
bounds = [(0,None)]*(K*E) + [(0, commodities[i]["requested_max"]) for i in range(K)]

# Equality constraints: flow conservation per commodity per node
A_eq_rows = []
b_eq = []
for ki,com in enumerate(commodities):
    s = com["src"]; t = com["dst"]
    for v in nodes:
        row = [0.0]*n_vars
        for ei,(u,vv) in enumerate(edges):
            if u == v:
                row[ki*E + ei] = 1.0
            if vv == v:
                row[ki*E + ei] -= 1.0
        x_idx = K*E + ki
        if v == s:
            row[x_idx] = -1.0
        elif v == t:
            row[x_idx] = 1.0
        A_eq_rows.append(row); b_eq.append(0.0)

# Inequality constraints
A_ub_rows = []; b_ub = []
# capacity constraints
for ei,(u,v) in enumerate(edges):
    row = [0.0]*n_vars
    for ki in range(K):
        row[ki*E + ei] = 1.0
    A_ub_rows.append(row); b_ub.append(G[u][v]["capacity"])
# delay constraints per commodity: sum_e d_e * f_k_e <= Dk * x_k
for ki,com in enumerate(commodities):
    row = [0.0]*n_vars
    for ei,(u,v) in enumerate(edges):
        row[ki*E + ei] = G[u][v]["delay"]
    row[K*E + ki] = -com["Dk"]
    A_ub_rows.append(row); b_ub.append(0.0)

A_eq = np.array(A_eq_rows); b_eq = np.array(b_eq)
A_ub = np.array(A_ub_rows); b_ub = np.array(b_ub)

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
success = res.success
print("LP solved:", success, "-", res.message)

x = res.x if success else np.zeros(n_vars)

# Extract allocations and flows
allocs = []
for ki,com in enumerate(commodities):
    xk = x[K*E + ki]
    allocs.append({"commodity":com["id"],"src":com["src"],"dst":com["dst"],"requested_max":com["requested_max"],
                   "revenue_per_Mbps":com["revenue"],"Dk_ms":com["Dk"],"x_allocated_Mbps":round(float(xk),4)})
allocs_df = pd.DataFrame(allocs)

# Edge utilizations
edge_rows = []
for ei,(u,v) in enumerate(edges):
    used = sum([x[ki*E + ei] for ki in range(K)])
    cap = G[u][v]["capacity"]
    edge_rows.append({"u":u,"v":v,"capacity_Mbps":cap,"utilized_Mbps":round(used,4),"utilization_frac":round(used/cap if cap>0 else 0,4),"delay_ms":G[u][v]["delay"]})
edge_df = pd.DataFrame(edge_rows).sort_values("utilization_frac", ascending=False)

# Path decomposition per commodity
paths_by_com = {}
for ki,com in enumerate(commodities):
    s = com["src"]; t = com["dst"]
    Gflow = nx.DiGraph()
    for ei,(u,v) in enumerate(edges):
        flow = x[ki*E + ei]
        if flow > 1e-8:
            Gflow.add_edge(u,v,flow=flow)
    simple_paths = []
    if s in Gflow.nodes() and t in Gflow.nodes():
        try:
            for path in nx.shortest_simple_paths(Gflow, source=s, target=t):
                bottleneck = min([Gflow[u][v]['flow'] for u,v in zip(path[:-1], path[1:])])
                simple_paths.append((path,bottleneck))
                for u,v in zip(path[:-1], path[1:]):
                    Gflow[u][v]['flow'] -= bottleneck
                    if Gflow[u][v]['flow'] <= 1e-9:
                        Gflow.remove_edge(u,v)
                if not Gflow.edges():
                    break
        except Exception:
            pass
    paths_by_com[com["id"]] = simple_paths

# Save outputs
os.makedirs("/mnt/data/abilene_sim_output", exist_ok=True)
allocs_df.to_csv("/mnt/data/abilene_sim_output/allocations.csv", index=False)
edge_df.to_csv("/mnt/data/abilene_sim_output/edge_utilization.csv", index=False)

# Plot network with utilization
pos = nx.spring_layout(G, seed=2)
plt.figure(figsize=(10,8))
util_vals = [edge_df[(edge_df.u==u)&(edge_df.v==v)]["utilization_frac"].values[0] for u,v in edges]
widths = [1 + 6*min(1.0,uv) for uv in util_vals]
nx.draw_networkx_nodes(G, pos, node_size=450, node_color='lightblue')
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=widths, arrowsize=12)
edge_labels = {(u,v): f"{edge_df[(edge_df.u==u)&(edge_df.v==v)]['utilized_Mbps'].values[0]:.1f}/{G[u][v]['capacity']}" for u,v in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Abilene topology - edge utilization (utilized/capacity in Mbps)")
plt.axis('off')
plt.tight_layout()
plt.savefig("/mnt/data/abilene_sim_output/abilene_network_utilization.png", dpi=200)
plt.show()

# Plot per-commodity paths
plt.figure(figsize=(10,8))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='whitesmoke')
nx.draw_networkx_labels(G, pos)
colors = plt.cm.get_cmap('tab10', K)
for idx,com in enumerate(commodities):
    for (path,bottleneck) in paths_by_com[com["id"]]:
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])), width=2+4*(bottleneck/max(1,com["requested_max"])), edge_color=[colors(idx)])
plt.title("Abilene - per-commodity path decomposition (colors = commodities)")
plt.axis('off')
plt.tight_layout()
plt.savefig("/mnt/data/abilene_sim_output/abilene_commodity_paths.png", dpi=200)
plt.show()

# Bar chart of allocations
plt.figure(figsize=(8,4))
plt.bar(allocs_df['commodity'], allocs_df['x_allocated_Mbps'], color='orange')
plt.xlabel("Commodity"); plt.ylabel("Allocated throughput (Mbps)"); plt.title("Allocated throughput per commodity (Abilene)")
plt.tight_layout()
plt.savefig("/mnt/data/abilene_sim_output/abilene_allocated_throughput.png", dpi=200)
plt.show()

# Display tables and summary
print("=== Edge table (simplified parameters) ===")
display(edge_df.reset_index(drop=True))
print("\n=== Commodity list and simplified parameters ===")
display(pd.DataFrame(commodities))
print("\n=== Allocations (results) ===")
display(allocs_df)

print("\nFiles saved to /mnt/data/abilene_sim_output/:")
for f in os.listdir("/mnt/data/abilene_sim_output"):
    print(" -", f)
