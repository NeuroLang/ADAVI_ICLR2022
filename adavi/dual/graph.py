import numpy as np

from typing import List, Tuple, Dict


def get_inference_order(
    graph: Tuple[Tuple[str, Tuple[str, ...]], ...]
) -> List[str]:
    """From a graph, derives an inference order so that
    at the time we calculate the value corresponding to a vertex all
    values for vertices it depends upon have already been calculated

    Parameters
    ----------
    graph : Tuple[Tuple[str, Tuple[str, ...]], ...]
        describes the graph as (rv, parents_of_rv)

    Returns
    -------
    List[str]
        order in which to process the rvs
    """
    rvs = [rv for rv, _ in graph]
    is_calculated = {rv: False for rv in rvs}
    inference_order = []
    while len(inference_order) < len(rvs):
        for rv, parents_of_rv in graph:
            if rv in inference_order:
                pass
            elif all(
                [
                    is_calculated[parent]
                    for parent in parents_of_rv
                ]
            ):
                inference_order.append(rv)
                is_calculated[rv] = True

    return inference_order


def ground_template(
    template_graph: Tuple[Tuple[str, Tuple[str, ...]], ...],
    plates_per_rv: Dict[str, List[str]],
    plate_cardinalities: Dict[str, int]
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    ground_graph = []
    template_per_rv = {}
    for rv, dependencies in template_graph:
        if len(plates_per_rv[rv]) > 0:
            dummy = np.empty(
                tuple(
                    plate_cardinalities[plate]
                    for plate in plates_per_rv[rv]
                )
            )
            for index, _ in np.ndenumerate(dummy):
                ground_rv = rv + "_".join(str(i) for i in index)
                ground_dependencies = []
                for dependency_rv in dependencies:
                    ground_dependency_rv = dependency_rv + "_".join(
                        str(index[i])
                        for i, plate in enumerate(plates_per_rv[rv])
                        if plate in plates_per_rv[dependency_rv]
                    )
                    ground_dependencies.append(
                        ground_dependency_rv
                    )
                ground_graph.append(
                    (ground_rv, tuple(ground_dependencies))
                )
                template_per_rv[ground_rv] = {
                    "template": rv,
                    "index": [*index]
                }
        else:
            ground_graph.append(
                (rv, dependencies)
            )
            template_per_rv[rv] = {
                "template": rv,
                "index": None
            }
    return tuple(ground_graph), template_per_rv
