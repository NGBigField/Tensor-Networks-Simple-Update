from kagome.src import triangle as triangle_lattice
from kagome.src import edges
from kagome.src._common import Node
from kagome.src import directions
from kagome.src.directions import LatticeDirection, BlockSide

from kagome.src._error_types import LatticeError, DirectionError
from kagome.src._types import EdgeIndicatorType

from kagome.src.utils import tuples, lists

## Oop:
from dataclasses import dataclass, fields
from typing import Generator

import itertools
import functools

## constants:

_delta_xs = [0, -1,  1]
_delta_ys = [1, -1, -1]

_CONSTANT_X_SHIFT = 3
_CONSTANT_y_SHIFT = 1

"""
An Upper Triagle:
     Up
      |
      |
      O
     / \
    /   \
Left    Right
"""

## Naming shortcuts:
L  = LatticeDirection.L 
R  = LatticeDirection.R 
UL = LatticeDirection.UL 
UR = LatticeDirection.UR 
DL = LatticeDirection.DL 
DR = LatticeDirection.DR


class KagomeLatticeError(LatticeError):...


@dataclass(frozen=False, slots=True)
class UpperTriangle:
    up    : Node = None
    left  : Node = None
    right : Node = None
    #
    index : int = -1

    def __getitem__(self, key:str)->Node:
        match key:
            case 'up': return self.up
            case 'left': return self.left
            case 'right': return self.right
            case _:
                raise KeyError(f"Not a valid key {key!r}")

    def all_nodes(self)->Generator[Node, None, None]:
        yield self.up
        yield self.left
        yield self.right
    
    @staticmethod
    def field_names()->list[str]:
        return ['up', 'left', 'right']

class _UnassignedEdgeName():
    def __repr__(self) -> str:
        return "_UnassignedEdgeName"
    


def num_message_connections(N:int)->int:
    return 2*N - 1


def edge_name_from_indices(i1:int, i2:int)->str:
    if   i1<i2:  return f"{i1}-{i2}" 
    elif i1>i2:  return f"{i2}-{i1}" 
    else:
        raise ValueError("Indices must be of different nodes") 


def _derive_node_directions(field:str)->list[LatticeDirection]:
    match field:
        case "up"   : return [LatticeDirection.UL, LatticeDirection.DL, LatticeDirection.DR, LatticeDirection.UR]
        case "left" : return [LatticeDirection.L, LatticeDirection.DL, LatticeDirection.R, LatticeDirection.UR]
        case "right": return [LatticeDirection.UL, LatticeDirection.L, LatticeDirection.DR, LatticeDirection.R]
        case _: raise ValueError(f"Unexpected string {field!r}")


def _tag_boundary_nodes(triangle:UpperTriangle, boundary:BlockSide)->None:
    touching_nodes : list[Node] = []
    if   boundary is BlockSide.U:     touching_nodes = [triangle.up]
    elif boundary is BlockSide.DL:    touching_nodes = [triangle.left]
    elif boundary is BlockSide.DR:    touching_nodes = [triangle.right]
    elif boundary is BlockSide.D:     touching_nodes = [triangle.left, triangle.right]
    elif boundary is BlockSide.UR:    touching_nodes = [triangle.up, triangle.right]
    elif boundary is BlockSide.UL:    touching_nodes = [triangle.up, triangle.left]
    else: 
        raise DirectionError()

    for node in touching_nodes:
        node.boundaries.add(boundary)


def get_upper_triangle_vertices_order(major_direction:BlockSide, minor_direction:LatticeDirection) -> list[list[str]]:
    match major_direction:
        case BlockSide.U:
            if   minor_direction is LatticeDirection.R:    return [['left', 'right'], ['up']]
            elif minor_direction is LatticeDirection.L:    return [['right', 'left'], ['up']]
            else: raise DirectionError("Impossible")
        case BlockSide.UR:
            if   minor_direction is LatticeDirection.DR:    return [['left'], ['up', 'right']]
            elif minor_direction is LatticeDirection.UL:    return [['left'], ['right', 'up']]
            else: raise DirectionError("Impossible")
        case BlockSide.UL:
            if   minor_direction is LatticeDirection.UR:    return [['right'], ['left', 'up']]
            elif minor_direction is LatticeDirection.DL:    return [['right'], ['up', 'left']]
            else: raise DirectionError("Impossible")
        case BlockSide.D:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.U, minor_direction))
        case BlockSide.DL:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.UR, minor_direction))
        case BlockSide.DR:
            return lists.reversed(get_upper_triangle_vertices_order(BlockSide.UL, minor_direction))

def _create_upper_triangle(triangular_node:Node, indices:list[int])->UpperTriangle:
    upper_triangle = UpperTriangle()
    x0, y0 = triangular_node.pos

    ## Derive Position and Directions:
    for node_index, field, delta_x, delta_y in zip(indices, UpperTriangle.field_names(), _delta_xs, _delta_ys, strict=True):
        x = x0 + delta_x + _CONSTANT_X_SHIFT
        y = y0 + delta_y + _CONSTANT_y_SHIFT
        node = Node(
            index=node_index,
            pos=(x, y),
            edges=[_UnassignedEdgeName(), _UnassignedEdgeName(), _UnassignedEdgeName(), _UnassignedEdgeName()],
            directions=_derive_node_directions(field)
        )
        upper_triangle.__setattr__(field, node)
    return upper_triangle
    
    
def _connect_kagome_nodes_inside_triangle(upper_triangle:UpperTriangle)->None:
        up, left, right = upper_triangle.up, upper_triangle.left, upper_triangle.right 
        # Up-Left:
        edge_name = edge_name_from_indices(up.index, left.index)
        up.edges[up.directions.index(DL)] = edge_name
        left.edges[left.directions.index(UR)] = edge_name
        # Up-Right:
        edge_name = edge_name_from_indices(up.index, right.index)
        up.edges[up.directions.index(DR)] = edge_name
        right.edges[right.directions.index(UL)] = edge_name
        # Left-Right:
        edge_name = edge_name_from_indices(left.index, right.index)
        left.edges[left.directions.index(R)] = edge_name
        right.edges[right.directions.index(L)] = edge_name   


def _name_outer_edges(node:Node, order_ind:int, boundary:BlockSide, kagome_lattice:list[Node], N:int)->None:
    upper_triangle = get_upper_triangle(node.index, kagome_lattice, N)
    _edge_name = lambda ind: f"{boundary}-{ind}"

    if boundary is BlockSide.D:     
        if   node is upper_triangle.left:   node.set_edge_in_direction(DL, _edge_name(order_ind))
        elif node is upper_triangle.right:  node.set_edge_in_direction(DR, _edge_name(order_ind))
        else:
            raise LatticeError()
        
    elif boundary is BlockSide.DR:    
        assert node is upper_triangle.right
        node.set_edge_in_direction(DR, _edge_name(2*order_ind))
        node.set_edge_in_direction(R, _edge_name(2*order_ind+1))

    elif boundary is BlockSide.UR:    
        if   node is upper_triangle.right:  node.set_edge_in_direction(R,  _edge_name(order_ind))
        elif node is upper_triangle.up:     node.set_edge_in_direction(UR, _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is BlockSide.U:     
        assert node is upper_triangle.up
        node.set_edge_in_direction(UR, _edge_name(2*order_ind))
        node.set_edge_in_direction(UL, _edge_name(2*order_ind+1))

    elif boundary is BlockSide.UL:    
        if   node is upper_triangle.up:     node.set_edge_in_direction(UL, _edge_name(order_ind))
        elif node is upper_triangle.left:   node.set_edge_in_direction(L,  _edge_name(order_ind))
        else: 
            raise LatticeError()

    elif boundary is BlockSide.DL:    
        assert node is upper_triangle.left
        node.set_edge_in_direction(L,  _edge_name(2*order_ind))
        node.set_edge_in_direction(DL, _edge_name(2*order_ind+1))

    else:   
        raise DirectionError("Not a possible hexagonal lattice direction")    


def _connect_kagome_nodes_between_triangles(triangle1:UpperTriangle, triangle2:UpperTriangle, direction1to2:LatticeDirection)->None:
    """ 
    Given two upper triangles `triangle1` and `triangle2`, 
    where the `triangle2` is in direction `direction1to2` relative to `triangle1`,
    find the relevant nodes, and assign common edge between them
    """

    ## Choose the two relevant nodes:
    if   direction1to2 is L:
        n1 = triangle1.left
        n2 = triangle2.right

    elif direction1to2 is DL:
        n1 = triangle1.left
        n2 = triangle2.up

    elif direction1to2 is DR:
        n1 = triangle1.right
        n2 = triangle2.up

    elif direction1to2 is R:
        n1 = triangle1.right
        n2 = triangle2.left

    elif direction1to2 is UR:
        n1 = triangle1.up
        n2 = triangle2.left

    elif direction1to2 is UL:
        n1 = triangle1.up
        n2 = triangle2.right

    else: 
        raise DirectionError(f"Impossible direction {direction1to2!r}")

    ## Assign proper edge name to them:
    edge_name = edge_name_from_indices(n1.index, n2.index)
    leg_index1 = n1.directions.index(direction1to2)
    leg_index2 = n2.directions.index(direction1to2.opposite())
    n1.edges[leg_index1] = edge_name
    n2.edges[leg_index2] = edge_name

def _sorted_boundary_nodes(nodes:list[Node], boundary:BlockSide)->list[Node]:
    # Get relevant nodes:
    boundary_nodes = [node for node in nodes if boundary in node.boundaries]

    # Choose sorting key:
    if   boundary is BlockSide.U:     sorting_key = lambda node: -node.pos[0]
    elif boundary is BlockSide.UR:    sorting_key = lambda node: +node.pos[1] 
    elif boundary is BlockSide.DR:    sorting_key = lambda node: +node.pos[1]
    elif boundary is BlockSide.UL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.DL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.D:     sorting_key = lambda node: +node.pos[0]
    else:
        raise DirectionError(f"Impossible direction {boundary!r}")

    # Sort:
    return sorted(boundary_nodes, key=sorting_key)


def get_upper_triangle(node_index:int, nodes:list[Node], N:int)->UpperTriangle:
    triangle_index = node_index//3
    up_index = triangle_index*3
    left_index, right_index = up_index+1, up_index+2
    return UpperTriangle(
        up    = nodes[up_index],
        left  = nodes[left_index],
        right = nodes[right_index]
    )


def create_kagome_lattice(
    N:int
)->tuple[
    list[Node],
    list[UpperTriangle]
]:

    ## Create the triangular lattice we're based on:
    original_triangular_lattice = triangle_lattice.create_triangle_lattice(N)
    triangular_lattice_of_upper_triangles : list[UpperTriangle] = []

    ## Position upper-triangles at each node of the kagome lattice:
    kagome_lattice : list[Node] = []
    crnt_kagome_index = 0
    for triangular_node in original_triangular_lattice:
        # Assign crnt indices for the triangle:
        indices = list(range(crnt_kagome_index, crnt_kagome_index+3))
        crnt_kagome_index += 3

        # Scale up the distance between nodes:
        triangular_node.pos = tuples.multiply(triangular_node.pos, (2,4))

        # Create triangle:
        upper_triangle = _create_upper_triangle(triangular_node, indices)
        upper_triangle.index = len(triangular_lattice_of_upper_triangles)
        kagome_lattice.extend(upper_triangle.all_nodes())
        triangular_lattice_of_upper_triangles.append(upper_triangle)
        
    ## Assign Inner edges within the triangle:
    for upper_triangle in triangular_lattice_of_upper_triangles:
        _connect_kagome_nodes_inside_triangle(upper_triangle)         

    ## Assign Edges between triangles:
    for index1, triangle1 in enumerate(triangular_lattice_of_upper_triangles):
        for index2, direction1 in triangle_lattice.all_neighbors(index1, N):
            triangle2 = triangular_lattice_of_upper_triangles[index2]
            _connect_kagome_nodes_between_triangles(triangle1, triangle2, direction1)

    ## Tag all nodes on boundary:
    for triangle in triangular_lattice_of_upper_triangles:
        on_boundaries = triangle_lattice.check_boundary_vertex(triangle.index, N)
        for boundary in on_boundaries:
            _tag_boundary_nodes(triangle, boundary)

    ## Use ordered nodes to name Outer Edges
    for boundary in BlockSide.all_in_counter_clockwise_order():
        sorted_nodes = _sorted_boundary_nodes(kagome_lattice, boundary)
        for i, node in enumerate(sorted_nodes):
            _name_outer_edges(node, i, boundary, kagome_lattice, N)
    # The bottom-left node is falsely on its DL leg, fix it:
    bottom_left_corner_node = _sorted_boundary_nodes(kagome_lattice, BlockSide.D)[0]
    bottom_left_corner_node.set_edge_in_direction(DL, f"{BlockSide.D}-0")

    ## Plot test:
    if False:
        from lattices._common import plot
        plot(kagome_lattice)
        plot(original_triangular_lattice, node_color="black", edge_style="y--", node_size=5)

    return kagome_lattice, triangular_lattice_of_upper_triangles


class KagomeLattice():
    __slots__ =  "N", "nodes", "triangles", "edges"
    
    def __init__(self, N:int) -> None:
        kagome_lattice, triangular_lattice_of_upper_triangles = create_kagome_lattice(N)
        self.N : int = N
        self.nodes     : list[Node] = kagome_lattice
        self.triangles : list[UpperTriangle]   = triangular_lattice_of_upper_triangles
        self.edges     : dict[str, tuple[int, int]] = edges.edges_dict_from_edges_list(
            [node.edges for node in kagome_lattice]
        )
                    
    # ================================================= #
    #|              Basic Derived Properties           |#
    # ================================================= #                    
    @property
    def num_message_connections(self)->int:
        return num_message_connections(self.N)
    
    @property
    def size(self)->int:
        return len(self.nodes)

    # ================================================= #
    #|              Geometry Functions                 |#
    # ================================================= #
    def num_boundary_nodes(self, boundary:BlockSide)->int:
        if boundary in [BlockSide.U, BlockSide.DR, BlockSide.DL]:
            return self.N
        elif boundary in [BlockSide.D, BlockSide.UR, BlockSide.UL]:
            return 2*self.N
        else:
            raise DirectionError("Not a possible boundary direction")

    @functools.cache
    def nodes_indices_rows_in_direction(self, major_direction:BlockSide, minor_direction:LatticeDirection)->list[list[int]]:
        ## Prepare basic data:
        N = self.N
        min_x, max_x, min_y, max_y = self.position_min_max()
        assert directions.check.is_orthogonal(major_direction, minor_direction)
        crnt_vertices_order = get_upper_triangle_vertices_order(major_direction, minor_direction)

        ## Get Upper-Triangles sorted in wanted direction:
        triangle_indices_in_order = triangle_lattice.verices_indices_rows_in_direction(N, major_direction, minor_direction)

        ## The resuts, are each row of upper-triangles, twice, taking the relevant node from the upper-triangle:
        list_of_rows = []
        for row in triangle_indices_in_order:    # Upper-Triangle order:
            for vertices_names in crnt_vertices_order:
                row_indices = self._row_in_direction(row, vertices_names)
                list_of_rows.append(row_indices)
        return list_of_rows


    @functools.cache
    def position_min_max(self)->tuple[int, ...]:
        min_x, max_x = lists.min_max([node.pos[0] for node in self.nodes])
        min_y, max_y = lists.min_max([node.pos[1] for node in self.nodes])
        return min_x, max_x, min_y, max_y

    def get_center_triangle(self)->UpperTriangle:
        index = triangle_lattice.get_center_vertex_index(self.N)
        return self.triangles[index]

    # ================================================= #
    #|            Retrieve Inner Objects               |#
    # ================================================= #

    def nodes_and_triangles(self)->Generator[tuple[Node, UpperTriangle, ], None, None]:
        triangles_repeated_3_times = itertools.chain.from_iterable(itertools.repeat(triangle, 3) for triangle in self.triangles)
        return zip(self.nodes, triangles_repeated_3_times, strict=True)
    
    
    def get_neighbor(self, node:Node, edge_or_dir:EdgeIndicatorType|LatticeDirection)->Node:
        if isinstance(edge_or_dir, EdgeIndicatorType):
            edge = edge_or_dir
        elif isinstance(edge_or_dir, Direction):
            edge = node.get_edge_in_direction(edge_or_dir)
        else:
            raise TypeError(f"Not an expected type {type(edge_or_dir)!r}")

        i1, i2 = self.edges[edge]
        assert i1!=i2
        if i1 == node.index:
            return self.nodes[i2]
        elif i2 == node.index:
            return self.nodes[i1]
        else:
            raise LatticeError("No neighbor")
        
    @functools.cache
    def sorted_boundary_nodes(self, boundary:BlockSide)->list[Node]:
        return _sorted_boundary_nodes(self.nodes, boundary)
    
    @functools.cache
    def sorted_boundary_edges(self, boundary:BlockSide)->list[EdgeIndicatorType]:
        # Basic info:
        num_boundary_nodes = self.num_boundary_nodes(boundary)
        boundary_nodes = self.sorted_boundary_nodes(boundary)
        assert len(boundary_nodes)==num_boundary_nodes

        # Logic of participating directions:
        participating_directions = boundary.matching_lattice_directions()

        # Logic of participating edges and nodes:
        omit_last_edge = num_boundary_nodes==self.N
        omit_last_node = not omit_last_edge
        
        # Get all edges in order:
        boundary_edges = []
        for _, is_last_node, node in lists.iterate_with_edge_indicators(boundary_nodes):
            if omit_last_node and is_last_node:
                break

            for _, is_last_direction, direction in lists.iterate_with_edge_indicators(participating_directions):
                if omit_last_edge and is_last_node and is_last_direction:
                    break

                if direction in node.directions:
                    boundary_edges.append(node.get_edge_in_direction(direction))

        assert self.num_message_connections == len(boundary_edges)
        return boundary_edges


    def _row_in_direction(self, triangle_indicse:list[int], triangle_keys:list[str]) -> list[int]:
        node_indices = []
        for triangle_index in triangle_indicse:
            triangle = self.triangles[triangle_index]
            for key in triangle_keys:
                node : Node = triangle.__getattribute__(key)
                node_indices.append(node.index)
        return node_indices
    
    # ================================================= #
    #|                    Visuals                      |#
    # ================================================= #
    def plot_triangles_lattice(self)->None:
        # Visuals import:
        from utils import visuals
        from matplotlib import pyplot as plt
        # basic data:
        N = self.N
        # Plot triangles:
        for upper_triangle in self.triangles:
            ind = upper_triangle.index
            i, j = triangle_lattice.get_vertex_coordinates(ind, N)
            x, y = triangle_lattice.get_node_position(i, j, N)
            plt.scatter(x, y, marker="^")
            plt.text(x, y, f" [{ind}]")
        
            
        

def main_test():
    from project_paths import add_base, add_scripts
    add_base()
    add_scripts()
    from scripts.build_tn import main_test
    main_test()


if __name__ == "__main__":
    main_test()