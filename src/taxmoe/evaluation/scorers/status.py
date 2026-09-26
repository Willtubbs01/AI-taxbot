def status_metrics(predicted, gold):
    return {'accuracy':float(predicted==gold),'predicted':predicted,'gold':gold}
