import networkx as nx

import matplotlib.pyplot as plt 


def hierarchy_pos(G, width=1.0, vert_gap=0.2, vert_loc=0):
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        roots = [list(G.nodes())[0]]
    pos = {}

    def _recurse(node, left, right, vloc):
        xcenter = (left + right) / 2
        pos[node] = (xcenter, vloc)
        children = list(G.successors(node))
        if children:
            step = (right - left) / len(children)
            for i, child in enumerate(children):
                _recurse(child,
                         left + i * step,
                         left + (i + 1) * step,
                         vloc - vert_gap)

    n_roots = len(roots)
    slice_width = width / n_roots
    for i, root in enumerate(roots):
        _recurse(root,
                 left=i * slice_width,
                 right=(i + 1) * slice_width,
                 vloc=vert_loc)
    return pos


def plot_dependency_graph(structure, kc_list, pmr_initial, student,
                          kc_colors, z1=0.2, z2=0.75):
    name_to_idx = {name: i for i, name in enumerate(kc_list)}
    G = nx.DiGraph()
    for parent, children in structure.items():
        for child in children:
            G.add_edge(parent, child)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = hierarchy_pos(G)

    # bordure = zone ZPD (rouge / vert / bleu)
    def zone_edge(pmr):
        if pmr < z1:
            return "#c0392b"
        elif pmr < z2:
            return "#27ae60"
        else:
            return "#2471a3"

    node_colors = []
    edge_colors = []
    labels = {}
    for name in G.nodes():
        idx = name_to_idx.get(name, None)
        pmr = pmr_initial.get(idx, 0.0) if idx is not None else 0.0
        # remplissage = couleur du KC (même que la timeline)
        node_colors.append(kc_colors.get(idx, "#cccccc"))
        edge_colors.append(zone_edge(pmr))
        labels[name] = f"KC {idx}\n{name}\nPMR={pmr:.2f}"

    plt.figure(figsize=(14, 9))
    nx.draw(G, pos,
            labels=labels,
            node_color=node_colors,
            edgecolors=edge_colors,
            linewidths=3,
            node_size=4000,
            font_size=8,
            font_weight="bold",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20)
    plt.title(f"Graphe de dépendance — Élève {student}\n"
              f"remplissage = couleur du KC | bordure rouge: PMR<{z1} | "
              f"verte: ZPD [{z1},{z2}] | bleue: PMR>={z2}",
              fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

