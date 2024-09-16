from torch_geometric.nn import MessagePassing

class CountCellTypesConv(MessagePassing):
    """
    Count celltypes of neighbors by message passing

    modified from the previous implementation to pass into hyperedges 
    """
    def __init__(self,  **kwargs):
        # Initialize with sum aggregation
        kwargs.setdefault('aggr', 'add')
        super(CountCellTypesConv, self).__init__(flow='source_to_target', node_dim=0, **kwargs)

        #super(CountCellTypesConv, self).__init__(aggr='add')

    def forward(self, x, hyperedge_index):
        num_nodes = x.size(0)


        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        return self.propagate(hyperedge_index, x=x, size=(num_nodes, num_edges))

    def message(self, x_j):
        return x_j