from networkx import clustering, eccentricity
from torch_geometric.transforms import Compose

from .networkx import NetworkXTransform


class Eccentricity(NetworkXTransform):

    def nx_transform(self, data):
        return eccentricity(data)


class ClusteringCoefficient(NetworkXTransform):

    def nx_transform(self, data):
        return clustering(data)


def get_transform(name):

    if name == "eccentricity":
        transform = Eccentricity()
    elif name == "clustering_coefficient":
        transform = ClusteringCoefficient()
    elif name == "scatter":
        transform = Compose([Eccentricity(), ClusteringCoefficient(cat=True)])
    else:
        raise NotImplementedError("Unknown transform %s" % name)
    return transform