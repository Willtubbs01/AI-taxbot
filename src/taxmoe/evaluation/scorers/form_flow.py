from .sets import set_metrics
def edge_metrics(predicted_edges,gold_edges):
    norm=lambda xs:[tuple(x) for x in (xs or [])]
    return set_metrics(norm(predicted_edges),norm(gold_edges))
