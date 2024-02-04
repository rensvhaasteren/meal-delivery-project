import numpy as np
from functools import cached_property

# noinspection PyUnresolvedReferences
from generator import Generator
from navigator import Navigator


# Load a navigator instance for Paris
nav = Navigator("data/paris_map.txt")


class Simulator:
    vehicle_times: np.ndarray[int]
    vehicle_locations: np.ndarray[float]

    __visited: set

    def __init__(self, year: int, month: int, day: int, **kwargs):
        """ Initialize a simulator for a full day on date [year]-[month]-[day]
        Accepts all keyword arguments that are accepted by Generator class to tune its behavior.
        Param [look_ahead=1800] defines in seconds how long before the pickup time of an order it gets visible.
        Param [pickup_radius=3000] defines in meters the radius in which a vehicle is able to observe orders.
        Param [scale_arrival_rates=1] can be used to linearly scale the number of arrivals during this day.
        Param [scale_visibility_duration=1] can be used to linearly scale the duration until an order disappears.
        Param [scale_traffic_intensity=1] can be used to linearly scale the traffic intensity during this day.
        Param [seed=0] can be used to obtain different scenario's for this same date.
        """
        self.__generator = Generator(year, month, day, **kwargs)

    def reset(self, n_vehicles: int, *, seed=0):
        """ Restarts the current scenario and sets vehicle starting times and locations
        Optionally, you can adjust the [seed=0] to start the vehicles at different locations.
        """
        self.vehicle_locations, self.vehicle_times = self.__generator.start(n_vehicles, seed)

        self.__visited = set()
        self.__clear_cache()

    def pickup(self, order: int, *, shortest_route="distance"):
        """ Performs a pickup action for the current vehicle.
        This vehicle will fully complete the delivery of the specified [order].
        Only after that, it becomes available again at the finish time and location
        Optionally, you can specify whether it should use time or distances to find routes.
        Returns a tuple with the profit made by this order and total distance traveled.
        """

        pickup_time = self.orders[order]["pickup_time"]
        restaurant = self.orders[order]["restaurant"]
        customer = self.orders[order]["customer"]
        fee = self.orders[order]["fee"]

        start, pickup, finish = nav.find_nodes([self.current_location, restaurant, customer])
        routes = nav.find_routes([start, pickup], [pickup, finish], weight=shortest_route).diagonal()
        times, distances = nav.find_route_lengths(routes)

        profit = fee + self.__generator.tip(order)
        traffic = self.__generator.traffic(self.current_time)
        waiting_time = max(pickup_time - self.current_time + times[0], 0)

        self.__current_time += round(traffic * sum(times)) + waiting_time
        self.__current_location = customer
        self.__visited.add(order)
        self.__clear_cache()

        return profit, sum(distances)

    def relocate(self, coord, *, shortest_route="distance"):
        """ Performs a relocate action for the current vehicle.
        This vehicle will relocate to the specified coordinate [coord].
        Only after that, it becomes available again at the arrival time and this location.
        Optionally, you can specify whether it should use time or distances to find routes.
        Returns a tuple with the profit made (=0) and total distance traveled.
        """

        start, finish = nav.find_nodes([self.current_location, coord])
        route = nav.find_routes(start, finish, weight=shortest_route)
        time, distance = nav.find_route_lengths(route)

        traffic = self.__generator.traffic(self.current_time)

        self.__current_time += round(traffic * time)
        self.__current_location = coord
        self.__clear_cache()

        return 0, distance

    def wait(self, seconds: int):
        """ Performs a pickup action for the current vehicle.
        This vehicle will be idled at its current location for the full duration of [seconds].
        Only after that, it becomes available again at same location.
        Returns a tuple with the profit made (=0) and total distance traveled (=0).
        """

        self.__current_time += seconds
        self.__clear_cache()

        return 0, 0

    @property
    def finished(self) -> bool:
        """ Returns whether the simulation has finished """
        return self.current_time > self.__generator.time_horizon

    @property
    def current_time(self) -> int:
        """ Returns the current time of the simulation """
        return self.__current_time

    @property
    def current_location(self) -> np.ndarray[float]:
        """ Returns the location of the currently regarded vehicle in the simulation """
        return self.__current_location

    @cached_property
    def current_vehicle(self) -> int:
        """ Returns the currently regarded vehicle in the simulation """
        return np.argmin(self.vehicle_times)

    @cached_property
    def orders(self) -> dict[int, dict[str]]:
        """ Returns a dict of orders that can be picked by the current vehicle at the current time and location"""
        observation = self.__generator.observe(self.current_time, self.current_location)
        return {i: self.__format_observation(*obs) for i, *obs in zip(*observation) if i not in self.__visited}

    @property
    def vehicle_orders(self):
        vehicle_orders = {}
        for v, loc in enumerate(self.vehicle_locations):
            observation = self.__generator.observe(self.current_time, loc)
            vehicle_orders[v] = {i: self.__format_observation(*obs) for i, *obs in zip(*observation) if i not in self.__visited}
        return vehicle_orders

    @staticmethod
    def __format_observation(pickup_time, restaurant, customer, fee):
        return {"pickup_time": pickup_time, "restaurant": restaurant, "customer": customer, "fee": fee}

    def __clear_cache(self):
        for attr in ["current_vehicle", "orders"]:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def __current_time(self) -> int:
        return self.vehicle_times[self.current_vehicle].item()

    @__current_time.setter
    def __current_time(self, time):
        self.vehicle_times[self.current_vehicle] = time

    @property
    def __current_location(self):
        return self.vehicle_locations[self.current_vehicle]

    @__current_location.setter
    def __current_location(self, coord):
        self.vehicle_locations[self.current_vehicle] = coord


if __name__ == "__main__":
    # Example usage of the Simulator class. Note that all distances are in meters and all times are in seconds.
    import datetime as dt

    # Create a simulator for a specific date
    sim = Simulator(2024, 1, 9)

    # Start the simulation and specify a number of considered vehicles
    sim.reset(n_vehicles=5)

    # You can observe the current vehicle for which you should make a decision, and it's current time and location
    print("Vehicle: %d is at time %s at coordinate %s and needs a new action\n" %
          (sim.current_vehicle, dt.timedelta(seconds=sim.current_time), sim.current_location))

    # You can observe the orders that are available for the current vehicle
    print("Available orders at the start:", *sim.orders.values(), sep="\n", end="\n" * 3)
    print("-" * 125)

    # You can do any of the three actions, you can decide to wait at your current location
    profit, distance = sim.wait(seconds=300)
    print("We let the vehicle travel for %d meters collection a profit of %.2f" % (distance, profit))

    # This leads to the following new situation
    print("Vehicle: %d is at time %s at coordinate %s and needs a new action\n" %
          (sim.current_vehicle, dt.timedelta(seconds=sim.current_time), sim.current_location))
    print("Available orders:", *sim.orders.items(), sep="\n", end="\n\n")
    print("-" * 125, end="\n\n")

    # Or you can decide to relocate to another location where you expect better options
    profit, distance = sim.relocate([48.820123,  2.345678])
    print("We let the vehicle travel for %d meters collection a profit of %.2f" % (distance, profit), end="\n\n")

    # Note that while the previously chosen relocation is busy, we now have to decide for another vehicle first.
    print("Vehicle: %d is at time %s at coordinate %s and needs a new action\n" %
          (sim.current_vehicle, dt.timedelta(seconds=sim.current_time), sim.current_location))
    print("Available orders:", *sim.orders.items(), sep="\n", end="\n\n")
    print("-" * 125, end="\n\n")

    # And of course, we can also decide to start the delivery of an order.
    profit, distance = sim.pickup(order=1)
    print("We let the vehicle travel for %d meters collection a profit of %.2f" % (distance, profit), end="\n\n")

    # That was a long trip, was it worth it or would another option have been better?

