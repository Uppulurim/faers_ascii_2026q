import pandas as pd
import numpy as np
import networkx as nx
import json

BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1"

def build_knowledge_graph():
    print("[Knowledge Graph] Loading signal data...")
    
    signals   = pd.read_csv(f"{BASE}/signals_all_drugs.csv")
    review    = pd.read_csv(f"{BASE}/review_log.csv")
    ebgm      = pd.read_csv(f"{BASE}/ebgm_results.csv")
    subgroup  = pd.read_csv(f"{BASE}/subgroup_results.csv")
    rxnorm    = pd.read_csv(f"{BASE}/rxnorm_cache.csv")

    accepted = review[review["review_status"] == "ACCEPTED"][["drug","event","ror","count"]]
    ebgm_sig = ebgm[ebgm["ebgm_signal"] == True][["drug","event","ebgm","eb05"]]

    G = nx.Graph()

    # Drug nodes
    for drug in accepted["drug"].unique():
        rxrow = rxnorm[rxnorm["original_name"] == drug]
        norm_name = rxrow["normalized_name"].values[0] if len(rxrow) > 0 else drug
        G.add_node(drug, node_type="drug", normalized_name=str(norm_name),
                   report_count=int(signals[signals["drug"]==drug]["count"].sum()))

    # Event nodes
    for event in accepted["event"].unique():
        G.add_node(event, node_type="event")

    # Subgroup nodes
    for _, row in subgroup.iterrows():
        try:
            age_dist = eval(row["age_group_distribution"]) if isinstance(row["age_group_distribution"], str) else {}
            for age_group, count in age_dist.items():
                if count > 0 and age_group not in G:
                    G.add_node(age_group, node_type="age_group")
        except Exception:
            pass
        try:
            country_dist = eval(row["top_countries"]) if isinstance(row["top_countries"], str) else {}
            top_country = list(country_dist.keys())[0] if country_dist else None
            if top_country and top_country not in G:
                G.add_node(top_country, node_type="country")
        except Exception:
            pass

    # Drug-Event edges
    for _, row in accepted.iterrows():
        if row["drug"] in G and row["event"] in G:
            ebgm_confirmed = len(ebgm_sig[
                (ebgm_sig["drug"] == row["drug"]) &
                (ebgm_sig["event"] == row["event"])
            ]) > 0
            ebgm_val = ebgm_sig[
                (ebgm_sig["drug"] == row["drug"]) &
                (ebgm_sig["event"] == row["event"])
            ]["ebgm"].values
            G.add_edge(row["drug"], row["event"],
                      edge_type="drug_event",
                      ror=float(row["ror"]),
                      count=int(row["count"]),
                      ebgm_confirmed=ebgm_confirmed,
                      ebgm=float(ebgm_val[0]) if len(ebgm_val) > 0 else None)

    # Drug-Subgroup edges
    for _, row in subgroup.iterrows():
        drug = row["drug"]
        if drug not in G:
            continue
        try:
            age_dist = eval(row["age_group_distribution"]) if isinstance(row["age_group_distribution"], str) else {}
            dominant_age = max(age_dist, key=age_dist.get) if age_dist else None
            if dominant_age and dominant_age in G and not G.has_edge(drug, dominant_age):
                G.add_edge(drug, dominant_age, edge_type="drug_subgroup",
                           subgroup_type="age_group", count=age_dist[dominant_age])
        except Exception:
            pass
        try:
            country_dist = eval(row["top_countries"]) if isinstance(row["top_countries"], str) else {}
            top_country = list(country_dist.keys())[0] if country_dist else None
            if top_country and top_country in G and not G.has_edge(drug, top_country):
                G.add_edge(drug, top_country, edge_type="drug_country",
                           count=list(country_dist.values())[0])
        except Exception:
            pass

    print(f"[Knowledge Graph] Built graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Drug nodes: {sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='drug')}")
    print(f"  Event nodes: {sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='event')}")
    return G

def analyze_graph(G):
    print("\n[Knowledge Graph] Graph Analysis:")

    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most connected nodes (degree centrality):")
    for node, score in top_nodes:
        node_type = G.nodes[node].get("node_type", "unknown")
        print(f"  [{node_type}] {node}: {score:.3f}")

    drug_nodes = [n for n,d in G.nodes(data=True) if d.get("node_type")=="drug"]
    print("\nDrug-Drug shared signal connections:")
    for i, d1 in enumerate(drug_nodes):
        for d2 in drug_nodes[i+1:]:
            neighbors1 = set(G.neighbors(d1))
            neighbors2 = set(G.neighbors(d2))
            shared = neighbors1 & neighbors2
            event_shared = [n for n in shared if G.nodes[n].get("node_type") == "event"]
            if event_shared:
                print(f"  {d1} <-> {d2}: {len(event_shared)} shared events")
                for e in event_shared[:3]:
                    print(f"    - {e}")

    components = list(nx.connected_components(G))
    print(f"\nConnected components: {len(components)}")
    for i, comp in enumerate(sorted(components, key=len, reverse=True)[:3]):
        print(f"  Component {i+1}: {len(comp)} nodes")

def export_graph(G):
    data = nx.node_link_data(G)
    json_path = f"{BASE}/knowledge_graph.json"
    with open(json_path, "w") as f:
        json.dump(data, f, default=str)
    print(f"\n[Knowledge Graph] Exported to {json_path}")

    # Clean None values before GEXF export
    for u, v, d in G.edges(data=True):
        for k, val in d.items():
            if val is None:
                G[u][v][k] = ""
    gexf_path = f"{BASE}/knowledge_graph.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"[Knowledge Graph] Exported to {gexf_path}")

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append(
            {
            "source": u, "target": v,
            "edge_type": d.get("edge_type",""),
            "ror": d.get("ror",""),
            "count": d.get("count",""),
            "ebgm_confirmed": d.get("ebgm_confirmed",""),
            "ebgm": d.get("ebgm","")
        })
    edge_df = pd.DataFrame(edges)
    edge_path = f"{BASE}/knowledge_graph_edges.csv"
    edge_df.to_csv(edge_path, index=False)
    print(f"[Knowledge Graph] Edge list saved to {edge_path}")

if __name__ == "__main__":
    G = build_knowledge_graph()
    analyze_graph(G)
    export_graph(G)
    print("\n[Knowledge Graph] Done.")
    print("Files written: knowledge_graph.json, knowledge_graph.gexf, knowledge_graph_edges.csv")