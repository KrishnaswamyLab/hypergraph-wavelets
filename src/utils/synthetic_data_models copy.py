import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import networkx as nx 
from sklearn.neighbors import kneighbors_graph
import time
from tqdm import trange
import graphtools as gt
from scipy.integrate import solve_ivp
import os
import sys 
# add the path to the sys path
from .spatial_patterning_funcs import gray_scott

class SyntheticSpatialData:
    def __init__(self, 
                 grid_mode, 
                 n_samples = None, 
                 sample_spacing=None, 
                 knn=5,
                 n_node_types = 6,
                 node_feature_dim = 100,
                 node_feature_noise = 0.01,
                 node_feature_gen_func = None,
                 feature_layout = 'rings',
                 feature_layout_dict = None):
        
        self.grid_mode = grid_mode
        self.n_samples = n_samples
        self.knn = knn
        self.sample_spacing = sample_spacing
        self.coords = self.gen_coords(grid_mode, n_samples, sample_spacing)
        #self.graph = self.gen_graph(self.coords, self.grid_mode, knn = self.knn, sample_spacing = self.sample_spacing)
        self.node_feature_dim = node_feature_dim
        self.n_node_types = n_node_types
        self.feature_layout = feature_layout
        self.feature_layout_dict = feature_layout_dict
        self.graph = None
        if node_feature_gen_func is None:
            self.node_feature_centers = np.random.normal(size=(n_node_types, node_feature_dim)) 
            # normalize the node_feature_centers to have unit norm
            self.node_feature_centers = self.node_feature_centers #/np.linalg.norm(self.node_feature_centers)
            self.node_feature_gen_func = lambda x: np.random.normal(self.node_feature_centers[x], node_feature_noise)
        self.data = None; self.cell_type_list = None
        # initialize self data and cell type list
        self.generate_feature_layout(self.feature_layout, self.feature_layout_dict)
    
    def plot(self):
        if self.cell_type_list is not None:
            # plot the coordinates with colors
            cell_types = np.unique(self.cell_type_list)
            for cell_type in cell_types:
                plt.scatter(self.coords[self.cell_type_list == cell_type, 0], self.coords[self.cell_type_list == cell_type, 1], label = cell_type)

            #plt.scatter(self.coords[:,0], self.coords[:,1], c = self.cell_type_list)
            # add in a legend for the colors
            plt.legend(loc='upper left')
        else:
            plt.scatter(self.coords[:,0], self.coords[:,1])
        # draw the edges of the graph
        if self.graph is not None:
            for i in range(self.graph.shape[0]):
                for j in range(self.graph.shape[1]):
                    if self.graph[i, j] == 1:
                        plt.plot([self.coords[i, 0], self.coords[j, 0]], [self.coords[i, 1], self.coords[j, 1]], 'k-', lw=0.5, alpha = 0.8)
        plt.show()

    def gen_coords(self, mode, n_samples = None, sample_spacing=None):
        # assume coordinates are on unit square
        if mode == 'grid':
            if sample_spacing is None:
                sample_spacing = 0.1
                self.sample_spacing = sample_spacing
            # generate grid coordinates
            x = np.arange(0, 1, sample_spacing)
            y = np.arange(0, 1, sample_spacing)
            xx, yy = np.meshgrid(x, y)
            coords = np.vstack([xx.ravel(), yy.ravel()]).T

        elif mode == 'random':
            if n_samples is None:
                n_samples = 1000
            coords = np.random.rand(n_samples, 2)
        elif mode == 'offset':
            if sample_spacing is None:
                sample_spacing = 0.1
                self.sample_spacing = sample_spacing
            x = np.arange(0, 1, sample_spacing)
            y = np.arange(0, 1, sample_spacing)
            # make a meshgrid but offset every other row by half the spacing
            xx, yy = np.meshgrid(x, y)
            xx[::2, :] += sample_spacing/2
            coords = np.vstack([xx.ravel(), yy.ravel()]).T
        else:
            raise ValueError('mode not recognized')
        # update self.num_samples
        self.n_samples = coords.shape[0]
        return coords

    def _gen_graph(self):
        # use graph tools to make the graph
        if self.knn is None:
            self.graph = gt.Graph(self.coords)
        else:
            self.graph = gt.Graph(self.coords, knn = self.knn, knn_max = self.knn)

    # def gen_graph(self, coords, mode, knn=None, sample_spacing=None):
    #     if mode == 'grid':
    #         # connect each point to its neighbors based on the grid structure
    #         # first make a graph
    #         if sample_spacing is None:
    #             sample_spacing = np.max(coords[0]-coords[1])
    #         A = np.zeros((coords.shape[0], coords.shape[0]))
    #         for i in trange(coords.shape[0]):
    #             for j in range(coords.shape[0]):
    #                 if np.linalg.norm(coords[i] - coords[j]) < sample_spacing*1.1 and i != j:
    #                     A[i, j] = 1
    #         return A
    #     elif mode == 'random':
    #         # make a knn graph from coords 
    #         assert knn is not None
    #         A = kneighbors_graph(coords, knn, mode='connectivity', include_self=False).toarray()
    #         return A
    #     elif mode == 'offset':
    #         # make a knn graph from coords 
    #         #A = kneighbors_graph(coords, 10, mode='connectivity', include_self=False).toarray()
    #         if sample_spacing is None:
    #             sample_spacing = np.max(coords[0]-coords[1])
    #         A = np.zeros((coords.shape[0], coords.shape[0]))
    #         for i in trange(coords.shape[0]):
    #             for j in range(coords.shape[0]):
    #                 if np.linalg.norm(coords[i] - coords[j]) < sample_spacing*1.4 and i != j:
    #                     A[i, j] = 1
    #         return A
    #     else:
    #         raise ValueError('mode not recognized')
        
    def generate_feature_layout(self, layout, layout_dict):
        if layout == 'rings':
            if layout_dict is None:
                layout_dict = {'n_rings': 3, 'ring_radius': 0.1, 'inner_radius': 0.05}
            return self._generate_features_rings(layout_dict['n_rings'], layout_dict['ring_radius'], layout_dict['inner_radius'])
        elif layout == 'random':
            if layout_dict is None:
                layout_dict = {'n_features': 2, 'n_classes': 3}
            return self._generate_features_random(layout_dict['n_features'], layout_dict['n_classes'])
        elif layout == 'reaction_diffusion':
            if layout_dict is None:
                layout_dict = {'equations': 'gray-scott', 'initializations': 'random', 'max_t': 10}
            return self._generate_features_reaction_diffusion(layout_dict['equations'], layout_dict['initializations'], max_t = layout_dict['max_t'])
        else:
            raise ValueError('layout not recognized')
        
    def _generate_features_reaction_diffusion(self, equations, initializations, max_t = 10, visualize=True):
        # ensure that the graph is given
        if self.graph is None:
            self._gen_graph()
        # initialize the data
        if initializations == 'random':
            u0 = np.random.rand(self.n_samples, )#self.node_feature_dim)
            v0 = np.random.rand(self.n_samples, )#self.node_feature_dim)
            # first half correspond to u 
            # second half correspond to v
        if equations== 'gray-scott':
            # Gray-Scott model
            # u_t = D_u \Delta u - uv^2 + F(1-u)
            # v_t = D_v \Delta v + uv^2 - (F+k)v
            # D_u = 0.16, D_v = 0.08, F = 0.035, k = 0.065
            D_u = 0.16; D_v = 0.08; F = 0.035; k = 0.065
            #dt = 0.01; dx = 0.1
            A = self.graph.K.toarray() > 0
            A = A ^ np.diag(np.diag(A))
            # convert to floats
            A = A.astype(float)
            L = np.diag(np.sum(A, axis=0)) - A

            y = np.concatenate((u0, v0))
            
            t_span = (0, max_t)
            t_eval = np.arange(0, max_t+.001, 1)
            #t_eval = np.arange(0, 100, 0.1)
            sol = solve_ivp(gray_scott, t_span, y, args=(L, D_u, D_v, F, k), t_eval = t_eval)
            import pdb; pdb.set_trace()
            
            for t_ind, t in enumerate(t_eval):
                y_sol = sol.y[:, t_ind]
                u = y_sol[:self.n_samples]
                v = y_sol[self.n_samples:]
                if visualize:
                    plt.plot()
                    plt.scatter(self.coords[:,0], self.coords[:,1], c = u[:, t_ind])
                    plt.title(f'u at time {t}')
                    plt.show()
                    plt.close()
                    plt.plot()
                    plt.scatter(self.coords[:,0], self.coords[:,1], c = v[:, t_ind])
                    plt.title(f'v at time {t}')
                    plt.show()
                    plt.close()


        self.data = np.zeros((self.n_samples, self.node_feature_dim))

    
    def _generate_features_random(self, n_features, n_classes, n_samples = None):
        if n_samples is None:
            n_samples = self.n_samples
        self.data = np.random.rand(n_samples, n_features)
        self.cell_type_list = np.random.randint(0, n_classes, n_samples)
        return self.data, self.cell_type_list
    
    def _generate_features_rings(self, n_rings, ring_radius, inner_radius):
        # choose background cell type 
        background_cell_type = np.random.randint(0, self.n_node_types)
        # choose ring center locations 
        ring_centers = np.random.rand(n_rings, 2)
        # choose ring cell types from the n_node_types except for the background_cell_type
        ring_cell_types_outer = np.random.randint(0, self.n_node_types-1, n_rings)
        ring_cell_types_outer[ring_cell_types_outer >= background_cell_type] += 1
        # inner radius cell type
        ring_cell_types_inner = np.random.randint(0, self.n_node_types-1, n_rings)
        ring_cell_types_inner[ring_cell_types_inner >= background_cell_type] += 1
        # make the data
        #self.data = np.zeros((self.n_samples, self.node_feature_dim))
        cell_type_list = []
        for i in range(self.n_samples):
            cell_type = background_cell_type
            for j in range(n_rings):
                if np.linalg.norm(self.coords[i] - ring_centers[j]) < inner_radius:
                    #self.data[i] = self.node_feature_gen_func(ring_cell_types_inner[j])
                    #print(f'inner ring cell type is {ring_cell_types_inner[j]}')
                    cell_type = ring_cell_types_inner[j]
                elif np.linalg.norm(self.coords[i] - ring_centers[j]) < ring_radius:
                    #self.data[i] = self.node_feature_gen_func(ring_cell_types_outer[j])
                    cell_type = ring_cell_types_outer[j]
                else:
                    #self.data[i] = self.node_feature_gen_func(background_cell_type)
                    #cell_type_list.append(background_cell_type)
                    pass
            
            cell_type_list.append(cell_type)
        self.cell_type_list = cell_type_list
        self.data = np.array([self.node_feature_gen_func(self.cell_type_list[i]) for i in range(self.n_samples)])


    
if __name__ == "__main__":
    # make a synthetic dataset
    data = SyntheticSpatialData('grid', sample_spacing = 0.1, feature_layout='reaction_diffusion')
    print(data.coords)
    print(data.graph)
    data = SyntheticSpatialData('random', n_samples = 1000, knn = 5)
    print(data.coords)
    print(data.graph)
    data = SyntheticSpatialData('offset', n_samples = 1000)
    print(data.coords)
    print(data.graph)
    # plot the coordinates
    plt.scatter(data.coords[:,0], data.coords[:,1])
    plt.show()
