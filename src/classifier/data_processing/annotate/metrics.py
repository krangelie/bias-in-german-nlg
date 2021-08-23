import pandas as pd
from nltk import agreement
from sklearn.metrics import cohen_kappa_score


def fleiss_kappa(data, annotator_names):
    formatted_codes = []

    for j, annotator in enumerate(annotator_names):
        formatted_codes += [[j, i, val] for i, val in enumerate(data[annotator])]

    ratingtask = agreement.AnnotationTask(data=formatted_codes)

    print("Fleiss' Kappa:", ratingtask.multi_kappa())


def get_all_pairwise_kappas(data, annotator_names, anonymize=True):
    a_names_cl = annotator_names
    if anonymize:
        annotator_names = [f"Annotator_{i}" for i, _ in enumerate(annotator_names)]
    results = pd.DataFrame()
    for i, a in enumerate(annotator_names):
        for j, b in enumerate(annotator_names):
            if j > i:
                results.loc[a, b] = cohen_kappa_score(
                    data[a_names_cl[i]], data[a_names_cl[j]]
                )
    print("Pairwise Cohen Kappa\n", results)
