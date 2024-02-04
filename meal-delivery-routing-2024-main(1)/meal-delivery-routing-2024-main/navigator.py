import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from cartopy.crs import PlateCarree
from cartopy.io.img_tiles import OSM
from collections import namedtuple
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

Node = namedtuple("node", ["lat", "lon"])
Edge = namedtuple("edge", ["start", "end", "directions", "time", "distance"])


class Navigator:
    def __init__(self, filename):
        """Initializes a navigator class based on a specifically formatted input file [filename]"""

        # Load network data from the file
        with open(filename) as file:
            n_nodes, n_edges, n_polygons = map(int, file.readline().split())
            self.nodes = [Node(*map(float, file.readline().split())) for _ in range(n_nodes)]
            self.edges = [Edge(*map(int, file.readline().split())) for _ in range(n_edges)]
            self.polygons = [Polygon(zip(*[map(float, file.readline().split()[::-1])] * 2)) for _ in range(n_polygons)]

        # Create array of node coordinates
        self.coords = np.array([[node.lat, node.lon] for node in self.nodes])

        # Create polygon of the entire area
        self.boundary = unary_union(self.polygons)

        # Make a new directed graph
        self.graph = nx.DiGraph()

        # Add all nodes and their data
        self.graph.add_nodes_from((i, {"coordinate": node}) for i, node in enumerate(self.nodes))

        # Add all edges and their data
        self.graph.add_edges_from(
            (
                edge.start, edge.end,
                {
                    "time": edge.time,          # In seconds
                    "distance": edge.distance   # In meters
                }
            )
            for edge in self.edges
        )
        self.graph.add_edges_from(
            (
                edge.end, edge.start,
                {
                    "time": edge.time,          # In seconds
                    "distance": edge.distance   # In meters
                }
            )
            for edge in self.edges if
            edge.directions == 2
        )

        # Initialize tiles loader
        self.__osm = OSM()

        # Create transformed coordinates on osm projection
        self.__projections = self.__osm.crs.transform_points(PlateCarree(), *self.coords.T[::-1])[:, :2]

        # Initialize subgraph cache subgraphs
        self.__subgraphs = {}

    def __contains__(self, coord):
        """ Check if a single (lat, lon) coordinate [coord] is within the boundary of the navigator"""

        return Point(*coord[::-1]).within(self.boundary)

    @staticmethod
    def haversine_distance(orgs, dests, *, radius=6371e3):
        """Find the direct haversine distance on a sphere with given [radius], by default the earth's radius in meters.
        If [orgs] and [dests] both are a single (lat, lon) coordinate, it will return a single distance.
        One or both of those can also be a sequence of coordinates to return an array or matrix respectively."""

        ndims = np.array([np.ndim(orgs), np.ndim(dests)])

        orgs = np.atleast_2d(orgs)
        dests = np.atleast_2d(dests)

        lat, lon = np.radians(orgs).T
        lat_, lon_ = np.radians(dests).T

        dlon = lon_ - lon[:, None]
        dlat = lat_ - lat[:, None]

        hav = np.sin(0.5 * dlat) ** 2 + np.cos(lat[:, None]) * np.cos(lat_) * np.sin(0.5 * dlon) ** 2
        dist = 2 * radius * np.arcsin(np.sqrt(hav))

        return Navigator.__squeeze(dist, axis=tuple(np.flatnonzero(ndims == 1)))

    def find_nodes(self, coords, *, chunk_size=None):
        """ Find the id of the node in the network with the smallest haversine distance to the given coordinate.
        If [coords] is a single (lat, lon) coordinate, it will return a single node id.
        This can also be a sequence of coordinates to return an array instead.
        For a large number of coordinates, specifying a [chunk_size] could be beneficial over a single operation."""

        if chunk_size is None:
            return np.argmin(Navigator.haversine_distance(self.coords, coords), axis=0)
        else:
            return np.concatenate([
                np.argmin(Navigator.haversine_distance(self.coords, coords[i:i+chunk_size]), axis=0)
                for i in tqdm(range(0, coords.shape[0], chunk_size))
            ])

    def find_routes(self, orgs, dests, *, weight="distance", max_speed=None):
        """Find the shortest [weight] path in the graph between nodes [orgs] and [dests]
        If [orgs] and [dests] both are a single node identifier, it will return a single route.
        One or both of those can also be a sequence of node identifiers to return an array or matrix respectively.
        Weight can be set to any edge attribute which are by default "distance" or "time".
        A [max_speed] can be set to disallow routes that contain edges where this speed would be exceeded."""

        ndims = np.array([np.ndim(orgs), np.ndim(dests)])

        orgs = np.atleast_1d(orgs)
        dests = np.atleast_1d(dests)

        getter = self.__itemgetter(items=dests)
        graph = self.__get_subgraph(max_speed)

        routes = np.empty((orgs.size, dests.size), object)
        routes[:] = [getter(nx.shortest_path(graph, org, weight=weight)) for org in orgs]

        return self.__squeeze(routes, axis=tuple(np.flatnonzero(ndims == 0)))

    def find_route_lengths(self, routes):
        """Find the total time and distance of a given route.
        If [routes] is a single route, it will return a tuple of two values.
        This can also be an array or matrix of routes to return a tuple of two arrays or matrices respectively.
        The returned tuple starts with the traveled time first, followed by traveled distance."""

        edge_times = self.__itemgetter(collection=nx.get_edge_attributes(self.graph, "time"))
        edge_distances = self.__itemgetter(collection=nx.get_edge_attributes(self.graph, "distance"))

        if np.ndim(routes[0]) == 0:
            route_times = sum(edge_times(zip(routes, routes[1:])))
            route_distances = sum(edge_distances(zip(routes, routes[1:])))
        else:
            route_times = np.vectorize(lambda route: sum(edge_times(zip(route, route[1:]))))(routes)
            route_distances = np.vectorize(lambda route: sum(edge_distances(zip(route, route[1:]))))(routes)

        return route_times, route_distances

    def display_network(self, *, max_speed=None, ax=None, figsize=(12, 8), **kwargs):
        """Plots the network on a given matplotlib [ax] or creates and returns one with the provided [figsize].
        Has an option to filter out edges that would exceed a certain [max_speed].
        Accepts all keyword arguments that are accepted by the networkx.draw_networkx method.
        E.g. node_size, node_color and node_shape as well as width, edge_color to style the plotted network."""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=self.__osm.crs))

        # Find node coordinates for plotting
        pos = dict(zip(range(len(self.nodes)), self.__projections))

        # Find subgraph for plotting
        graph = self.__get_subgraph(max_speed)

        # Plot all nodes and edges
        nx.draw_networkx(
            graph, ax=ax, pos=pos,
            with_labels=False,
            arrows=False,
            node_size=kwargs.pop("node_size", 0.1),
            node_color=kwargs.pop("node_color", ("gray", 0.5)),
            width=kwargs.pop("width", 0.5),
            edge_color=kwargs.pop("edge_color", ("black", 0.5)),
            **kwargs
        )

        return ax

    def display_nodes(self, nodes, *, ax=None, figsize=(12, 8), **kwargs):
        """Plots nodes on a given matplotlib [ax] or creates and returns one with the provided [figsize].
        [nodes] should either be a single node identifier present in the network or a sequence of those.
        Accepts all keyword arguments that are accepted by the networkx.draw_networkx_nodes method.
        E.g. node_size, node_color and node_shape to style the plotted nodes."""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=self.__osm.crs))

        # Find projected node coordinates for plotting
        pos = dict(zip(range(len(self.nodes)), self.__projections))

        # Plot given nodes in the given style
        zorder = kwargs.pop("zorder", 4)

        points = nx.draw_networkx_nodes(
            self.graph,
            ax=ax,
            pos=pos,
            nodelist=np.atleast_1d(nodes),
            node_size=kwargs.pop("node_size", 15),
            node_color=kwargs.pop("node_color", "blue"),
            node_shape=kwargs.pop("node_shape", "."),
            **kwargs
        )

        points.set_zorder(zorder)

        return ax

    def display_route(self, route, *, ax=None, figsize=(12, 8), **kwargs):
        """Plots route of nodes on a given matplotlib [ax] or creates and returns one with the provided [figsize].
        A single [route] should be provided, containing a sequence of node identifiers that are present in the network.
        Accepts all keyword arguments that are accepted by the networkx.draw_networkx_edges method.
        E.g. width, edge_color and arrows to style the plotted edges."""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=self.__osm.crs))

        # Find projected node coordinates for plotting
        pos = dict(zip(range(len(self.nodes)), self.__projections))

        # Plot given route in the given style
        zorder = kwargs.pop("zorder", 3)

        edges = nx.draw_networkx_edges(
            self.graph.edge_subgraph(zip(route, route[1:])),
            ax=ax,
            pos=pos,
            width=kwargs.pop("width", 2),
            arrows=kwargs.pop("arrows", False),
            edge_color=kwargs.pop("edge_color", "red"),
            **kwargs
        )

        edges.set_zorder(zorder)

        return ax

    def display_area(self, polygon, *, ax=None, figsize=(12, 8), **kwargs):
        """Plots an area on a given matplotlib [ax] or creates and returns one with the provided [figsize].
        A single shapely [polygon] should be provided, specifying the area to be plotted.
        Accepts all keyword arguments that are accepted by the matplotlib.artist.Artist class.
        E.g. face_color and edge_color to style the plotted area."""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=self.__osm.crs))

        ax.add_geometries(
            polygon,
            crs=PlateCarree(),
            facecolor=kwargs.pop("facecolor", ("green", 0.1)),
            edgecolor=kwargs.pop("edgecolor", ("green", 0.9)),
            zorder=kwargs.pop("zorder", 2),
            **kwargs
        )

        return ax

    def display_map(
            self,
            zoom=14,
            *,
            ax=None,
            figsize=(12, 8),
            center=True,
            grid=True,
            inner_color=('gray', 0.50),
            outer_color=('black', 0.50)
    ):
        """Adds OpenStreetMap background tiles to a given matplotlib [ax] or creates one with the provided [figsize].
        You can set a desired [zoom] level at which the tiles should be rendered.
        Optionally you can decide to [center] the map based on the network boundary and whether to add a [grid].
        The [inner_color] and [outer_color] specify the color of the overlays respectively in- and outside the boundary.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=self.__osm.crs))

        ax.add_image(self.__osm, zoom)

        if center:
            xmin, ymin, xmax, ymax = self.boundary.bounds
            ax.set_extent((xmin, xmax, ymin, ymax), crs=PlateCarree())

        if grid:
            ax.gridlines(draw_labels=True)

        ax.add_geometries(self.boundary, crs=PlateCarree(), facecolor=inner_color, zorder=1)
        ax.add_geometries(self.boundary.envelope.difference(self.boundary), crs=PlateCarree(), facecolor=outer_color)

        return ax

    @staticmethod
    def __squeeze(array, axis=None):
        squeezed = np.squeeze(array, axis=axis)
        return squeezed if squeezed.shape else squeezed.item()

    @staticmethod
    def __itemgetter(*, items=None, collection=None):
        if collection is None:
            def getter(collection):
                return [collection.__getitem__(item) for item in items]
            return getter

        if items is None:
            def getter(items):
                return [collection.__getitem__(item) for item in items]
            return getter

        return [collection.__getitem__(item) for item in items]

    def __get_subgraph(self, max_speed=None):
        if max_speed is None:
            return self.graph

        if max_speed not in self.__subgraphs:
            edges = ((i, j) for i, j, e in self.graph.edges(data=True) if e["distance"] / e["time"] <= max_speed)
            self.__subgraphs[max_speed] = self.graph.edge_subgraph(edges)

        return self.__subgraphs[max_speed]


if __name__ == "__main__":
    # Example usage of the Navigator class. Note that all distances are in meters and all times are in seconds.

    # Load the navigator class for the paris network file
    nav = Navigator("data/paris_map.txt")

    # # Define some example coordinates in the Paris city center
    # notre_dame = (48.85308818278138, 2.3499879275454667)
    # eiffel_tower = (48.85843706149484, 2.294489560459597)
    # louvre_museum = (48.86070981531768, 2.3376439796670487)
    # arc_de_triomphe = (48.87389046320069, 2.2950060391865157)

    # # Find the node identifiers corresponding to these coordinates
    # notre_dame = nav.find_nodes(notre_dame)
    # arc_de_triomphe = nav.find_nodes(arc_de_triomphe)

    # # Or by using a more efficient vectorized approach
    # eiffel_tower, louvre_museum = nav.find_nodes([eiffel_tower, louvre_museum])

    # # Select one of the regions in the city center, e.g. the one around the Eiffel Tower
     
    area = nav.polygons[1]
    print('HIER')
    print(area)
    polygon = Polygon(area)

    # Find the centroid (middle point) of the polygon
    centroid_point = polygon.centroid

    # Extract the centroid coordinates
    centroid_coords = (centroid_point.x, centroid_point.y)

    # Print or use the centroid coordinates
    print("Centroid Coordinates:", centroid_coords)

    # # Find the shortest route from the Arc De Triomphe to the Notre Dame
    # route = nav.find_routes(arc_de_triomphe, notre_dame)

    # # Find the corresponding traveled time and distance for this route
    # print("Time: %d seconds; Distance: %d meters" % nav.find_route_lengths(route))

    # # Create a plot and add any combination of elements you want to show

    # # Add background map tiles
    # ax = nav.display_map(ax=None)
    # plt.show(block=False)
    # plt.pause(3)

    # # Start with the network representation of the roads
    # ax = nav.display_network(ax=ax)
    # plt.show(block=False)
    # plt.pause(3)

    # # Add a number of nodes to the plot
    # ax = nav.display_nodes(notre_dame, ax=ax)
    # ax = nav.display_nodes(eiffel_tower, ax=ax)
    # ax = nav.display_nodes([arc_de_triomphe, notre_dame], ax=ax, node_color="gold", node_shape="*")
    # plt.show(block=False)
    # plt.pause(3)

    # # Add the route and area you found
    # ax = nav.display_route(route, ax=ax)
    # ax = nav.display_area(area, ax=ax)
    # plt.show()



