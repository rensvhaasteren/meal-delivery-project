from collections import defaultdict
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import math
import datetime as dt
'''
Create a class for the restaurant locations
'''
class Restaurant:
    def __init__(self, x, y):
        self.x = x
        self.y = y

'''
Class for the location of the customer
'''
class Customer:
    def __init__(self, x, y):
        self.x = x
        self.y = y

'''
This function will compare the coordinates of two objects. If the coordinate of both
objects are the same the function will return true, otherwise it will return false. 
'''
def same_coordinate(object1, object2):
    if object1.x == object2.x:
        if object1.y == object2.y:
            return True
    return False

'''
This function will read the orders and save information in the restaurant_dict.
The order list will keep track If the order has already been read. 
'''
def read_orders_V1(order_dictionary, order_list, restaurant_dict):
    for order_num, order_info in order_dictionary.items():
        if order_num not in order_list:
            restaurant_dict = read_restaurant_V1(order_info["restaurant"], restaurant_dict, order_info["fee"])

'''
This function will read the orders and save information in the restaurant_dict.
The order list will keep track If the order has already been read. 
'''
def read_orders_V2(order_dictionary, order_list, restaurant_dict, day_of_the_week, nav):
    for order_num, order_info in order_dictionary.items():
        if order_num not in order_list:
            order_list.append(order_num)
            restaurant_dict = read_restaurant_V2(order_info["restaurant"], restaurant_dict, order_info["fee"],
                                                 day_of_the_week, nav)

'''
This function will read the coordinate of a customer. It will update the number of
orders of the customer. 
'''
def read_customer(coordinate_array, customer_dict):
    new_customer = Customer(coordinate_array[0], coordinate_array[1])
    if customer_dict and customer_dict.items():
        for customer_number, customer_information in customer_dict.items():
            if same_coordinate(new_customer, customer_information["object_"]):
                customer_information["orders"] += 1
                break
        else:
            customer_dict[customer_number + 1] = {"object_": new_customer, "orders": 1}
    else:
        customer_dict[0] = {"object_": new_customer, "orders": 1}

'''
This function will read the coordinate of a restaurant. It will update the number 
of orders of the restaurant and the occurence of a certain fee.
'''
def read_restaurant_V1(coordinate_array, restaurant_dict, fee):
    new_restaurant = Restaurant(coordinate_array[0], coordinate_array[1])
    if restaurant_dict and restaurant_dict.items():
        for restaurant_number, restaurant_information in restaurant_dict.items():
            if same_coordinate(new_restaurant, restaurant_information["object_"]):
                restaurant_information["orders"] += 1
                for fee_value, fee_occurence in restaurant_information["fees"].items():
                    if fee_value == fee:
                        restaurant_information["fees"][fee_value] += 1
                        break
                else:
                    restaurant_information["fees"][fee] = 1
                break
        else:
            restaurant_dict[restaurant_number + 1] = {"object_": new_restaurant, "orders": 1, "fees": {fee: 1}}
    else:
        restaurant_dict[0] = {"object_": new_restaurant, "orders": 1, "fees": {fee: 1}}

'''
This function will read the coordinate of a restaurant. It will update the number 
of orders of the restaurant and the occurence of a certain fee for a specific day.
'''
def read_restaurant_V2(coordinate_array, restaurant_dict, fee, day_of_the_week, nav):
    new_restaurant = Restaurant(coordinate_array[0], coordinate_array[1])
    new_restaurant_node = str(nav.find_nodes(coordinate_array))
    if restaurant_dict and restaurant_dict.items():
        for restaurant_information in restaurant_dict.values():
            if same_coordinate(new_restaurant, restaurant_information["object_"]):
                restaurant_information["Day"][day_of_the_week]["orders"] += 1
                for fee_value, fee_occurence in restaurant_information["Day"][day_of_the_week]["fees"].items():
                    if fee_value == fee:
                        restaurant_information["Day"][day_of_the_week]["fees"][fee_value] += 1
                        break
                else:
                    restaurant_information["Day"][day_of_the_week]["fees"][fee] = 1
                break
        else:
            days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            restaurant_dict[new_restaurant_node] = {
                "object_": new_restaurant,
                "Day": {day: {"orders": 0, "fees": dict()} for day in days_of_the_week}
            }
            restaurant_dict[new_restaurant_node]["Day"][day_of_the_week]["orders"] = 1
            restaurant_dict[new_restaurant_node]["Day"][day_of_the_week]["fees"][fee] = 1
    else:
        days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        restaurant_dict[new_restaurant_node] = {
            "object_": new_restaurant,
            "Day": {day: {"orders": 0, "fees": dict()} for day in days_of_the_week}
        }
        restaurant_dict[new_restaurant_node]["Day"][day_of_the_week]["orders"] = 1
        restaurant_dict[new_restaurant_node]["Day"][day_of_the_week]["fees"][fee] = 1

def serialize_customer(obj):
    if isinstance(obj, Customer):
        return {"x": obj.x, "y": obj.y}
    raise TypeError("Object not serializable")


def serialize_restaurant(obj):
    if isinstance(obj, Restaurant):
        return {"x": obj.x, "y": obj.y}
    raise TypeError("Object not serializable")

'''
This function returns the list location for a specific day
'''
def day_to_position(day):
    if day == "Monday":
        return 0
    elif day == "Tuesday":
        return 1
    elif day == "Wednesday":
        return 2
    elif day == "Thursday":
        return 3
    elif day == "Friday":
        return 4
    elif day == "Saturday":
        return 5
    else:
        return 6

'''
This function will calculate the weighted average fee and save this in restaurant
dictionary.
'''
def calculating_average_fee_V1(restaurant_dict):
    for restaurant_number, restaurant_info in restaurant_dict.items():
        weighted_sum = 0
        for fee_value, fee_occurence in restaurant_info["fees"].items():
            weighted_sum += (float(fee_value) * fee_occurence)
        weighted_average_fee = weighted_sum / restaurant_info["orders"]
        restaurant_info["weighted_average_fee"] = weighted_average_fee

'''
This function will calculate and return the total orders in the dictionary
'''
def total_orders(dictionary):
    total_orders = 0
    for dict_key, dict_value in dictionary.items():
        total_orders += dict_value["orders"]
    return total_orders

'''
This function will calculate and return the total weighted
average fee in the dictionary.
'''
def total_weighted_average_fee(restaurant_dict):
    total_weighted_average_fee = 0
    for rest_info in restaurant_dict.values():
        total_weighted_average_fee += rest_info["weighted_average_fee"]
    return total_weighted_average_fee

'''
This function will calculate and return the total orders in the dictionary
for a specifc day.
'''
def total_orders_on_day(restaurant_dict, day):
    total_orders = 0
    for rest_info in restaurant_dict.values():
        total_orders += rest_info["Day"][day]["orders"]
    return total_orders

'''
This function will calculate and return the total weighted
average fee in the dictionary for a specific day.
'''
def total_weighted_average_fee_on_day(restaurant_dict, day):
    total_weighted_average_fee = 0
    for rest_info in restaurant_dict.values():
        total_weighted_average_fee += rest_info["Day"][day]["weighted_average_fee"]
    return total_weighted_average_fee

'''
This function will calculate percentage of orders and additionaly if the restaurant
dictionary is given the percentage of fees covered by each restaurant. It will
also update the dictionary
'''
def calculating_percentages_V1(dictionary):
    for dict_key, dict_value in dictionary.items():
        dictionary[dict_key]['percentage_orders'] = dictionary[dict_key]["orders"] / total_orders(dictionary)
        # If we have the restaurnt_dict:
        if len(dictionary[dict_key]) > 4:
            dictionary[dict_key]["percentage_fees"] = (
                        dictionary[dict_key]["weighted_average_fee"] / total_weighted_average_fee(dictionary))

'''
This function will calculate percentage of orders and the percentage of fees
covered by each restaurant on each day. 
'''
def calculating_percentages_V2(restaurant_dict):
    days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days_of_the_week:
        for rest_node, rest_info in restaurant_dict.items():
            restaurant_dict[rest_node]["Day"][day]['percentage_orders'] = restaurant_dict[rest_node]["Day"][day][
                                                                              "orders"] / total_orders_on_day(
                restaurant_dict, day)
            restaurant_dict[rest_node]["Day"][day]["percentage_fees"] = restaurant_dict[rest_node]["Day"][day][
                                    "weighted_average_fee"] / total_weighted_average_fee_on_day(restaurant_dict, day)

'''
Import a json file 
'''
def import_json_file(file_path):
    with open(file_path, 'r') as file:
        restaurant_json = json.load(file)
    return restaurant_json

'''
Export a json file
'''
def export_json_file(dictionary, file_path, seralize=False):
    if seralize == False:
        json_data = json.dumps(dictionary, indent=2)
        with open(file_path, 'w') as file:
            file.write(json_data)
    else:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=2, default=serialize_restaurant)

'''
Import a pickle file to be able to display it
'''
def import_pickle_file(file_path):
    with open(file_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

'''
Export a pickle file
'''
def export_pickle_file(dictionary, file_path):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(dictionary, pickle_file)


'''
This function will search for a specific node the nearest restaurant based on time.
It will return a list of lists of nearest restaurants, example: [[1234, 23],...].
First number in each list is the time from the node to the restaurant and the second number
is the restaurant number. 
'''
def finding_nearest_restaurant_V1(nav, restaurant_data, node_num):
    number_restaurants = 5
    travel_times = []
    for restaurant_num, restaurant_info in restaurant_data.items():
        restaurant_location = [restaurant_info['object_']['x'], restaurant_info['object_']['y']]
        restaurant_loc = nav.find_nodes(restaurant_location)
        route = nav.find_routes(node_num, restaurant_loc, weight="time")
        time, distance = nav.find_route_lengths(route)
        travel_times.append([time, restaurant_num])
    nearest_restaurants = sorted(travel_times, key=lambda x: x[0])[0:number_restaurants]
    return nearest_restaurants


'''
This function will search for a specific node the nearest restaurant that is 
open for a specific day based on time.
It will return a list of lists of nearest restaurants, example: [[1234, 23],...].
First number in each list is the time from the node to the restaurant and the
second number is the restaurant number. 
'''
def finding_nearest_restaurant_V2(nav, restaurant_data, node_num, day):
    number_restaurants = 5
    travel_times = []
    for rest_node, rest_info in restaurant_data.items():
        if rest_info["operating days"][day_to_position(day)] == 1:
            route = nav.find_routes(node_num, int(rest_node), weight="time")
            time, distance = nav.find_route_lengths(route)
            travel_times.append([time, rest_node])
    nearest_restaurants = sorted(travel_times, key=lambda x: x[0])[0:number_restaurants]
    return nearest_restaurants

'''
This function will calculate for each node in Paris the value of F(end_location).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)), where
X is the percentage of total order covered in this end location and 
Y is the percentage of the max weighted fee covered.
It will return a dictionary that contains for every end_location, the value F(end_location).
'''
def heat_map_nodes_V1(restaurant_data, nav):
    heat_map = dict()
    # Paris containts 11348 nodes
    for node in range(0, 11348):
        print(node)
        nearest_restaurants = finding_nearest_restaurant_V1(nav, restaurant_data, node)
        percentage_order = 0
        percentage_fee = 0
        total_time_nearest_restaurants = 0
        for restaurant in nearest_restaurants:
            percentage_order += restaurant_data[restaurant[1]]['percentage_orders']
            percentage_fee += restaurant_data[restaurant[1]]['percentage_fees']
            total_time_nearest_restaurants += restaurant[0]
        node_reward = (1 + percentage_order) * (1 + percentage_fee) * (1 / (1 + total_time_nearest_restaurants / 10000))
        heat_map[node] = node_reward
    return heat_map

'''
This function will calculate for each node in Paris the value of F(end_location).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)), where
X is the percentage of total order covered in this end location and 
Y is the percentage of the max weighted fee covered.
It will return a dictionary that contains for every end_location, the value F(end_location).
'''
def heat_map_nodes_V2(restaurant_data, nav):
    days_of_the_week = ["Tuesday", "Sunday"]
    heat_map = {day: dict() for day in days_of_the_week}
    for day in days_of_the_week:
        print(day)
        # Paris containts 11348 nodes
        for node in range(0, 11348):
            print(node)
            nearest_restaurants = finding_nearest_restaurant_V2(nav, restaurant_data, node, day)
            percentage_order = 0
            percentage_fee = 0
            total_time_nearest_restaurants = 0
            for restaurant in nearest_restaurants:
                percentage_order += restaurant_data[restaurant[1]]["Day"][day]['percentage_orders']
                percentage_fee += restaurant_data[restaurant[1]]["Day"][day]['percentage_fees']
                total_time_nearest_restaurants += restaurant[0]
            node_reward = (1 + percentage_order) * (1 + percentage_fee) * (
                        1 / (1 + total_time_nearest_restaurants / 10000))
            heat_map[day][node] = node_reward
    return heat_map

'''
This function determines the 
'''
def read_order_duration(order_dic, memory_dict, sim, day, month):
    for order_num in order_dic.keys():
        if (month, day) not in memory_dict.keys():
            memory_dict[(month, day)] = {order_num: [sim.current_time, sim.current_time]}
        elif order_num not in memory_dict[(month, day)].keys():
            memory_dict[(month, day)][order_num] = [sim.current_time, sim.current_time]
        else:
            memory_dict[(month, day)][order_num][1] = sim.current_time


def calculate_order_duration(order_dic):
    for date, orders in order_dic.items():
        for order_num, order_info in orders.items():
            duration = order_info[1] - order_info[0]
            order_dic[date][order_num] = [duration]

'''
Determine the area from a x and y coordinate
'''
def find_area(coordinates, nav):
    coordinates_transform = [coordinates[1], coordinates[0]]
    point = Point(coordinates_transform)
    for i in range(0, 330):
        area = nav.polygons[i]
        if area.contains(point):
            return i
    return None

'''
This function shows a plot of all restaurant locations
'''
def show_restaurant_locations(nav):
    res_data = import_json_file("data/restaurant_data_v2.json")
    res_data2 = dict()
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    bins = []
    for i in range(1,49,1):
        bins.append(500*i)
    
    for res_num, res_info in res_data.items():
        total_orders = 0
        for day in days_of_week:
            total_orders += res_info["Day"][day]["orders"]
        res_data2[res_num] = total_orders
    
    for res_num, res_size in res_data2.items():
        count = 1
        for max_limit in bins:
            if res_size < max_limit:    
                ax = nav.display_nodes(int(res_num), ax=ax, node_color="black", node_shape=".", node_size=40*count)
                break
            count += 1
    plt.show()

'''
The function show_heatmap() shows a plot of the heatmap_v1
'''
def show_heatmap(nav):
    heatmap = import_json_file("data/heatmap_v1.json")
    
    area_values = defaultdict(list)
    
    for node in heatmap.keys():
        coordinates = [nav.nodes[int(node)][0],nav.nodes[int(node)][1]]
        area = find_area(coordinates, nav)
        if area != None:
            area_values[area].append(heatmap[node])
    
    area_values2 = dict()
    for area, values in area_values.items():
        area_values2[area] = np.mean(values)  
    
    diff = max(area_values2.values())-min(area_values2.values())
    steps = diff/9
    
    bins = []
    for i in range(1,10):
        value = min(area_values2.values()) + steps*i
        if i == 9:
            bins.append(2)
        else:
            bins.append(value)
    
    for area, value in area_values2.items():
        count = 1
        for max_limit in bins:
            if value < max_limit:
                if count <= 3:
                    if count == 1:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("red", 0.6), edgecolor=("red", 0.6))
                    elif count == 2:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("red", 0.4), edgecolor=("red", 0.6))
                    else:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("red", 0.2), edgecolor=("red", 0.6))
                elif count <= 6:
                    ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("orange", 0.4), edgecolor=("orange", 0.6))
                else:
                    if count == 7:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("green", 0.2), edgecolor=("green", 0.6))
                    elif count == 8:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("green", 0.4), edgecolor=("green", 0.6))
                    else:
                        ax = nav.display_area(nav.polygons[area], ax=ax, facecolor=("green", 0.6), edgecolor=("green", 0.6))
                break
            count += 1
    plt.show()

'''
This function returns a list of list of nearest restaurants. This is based on the
time from the customer to the restaurant. This version use customer coordinate.
'''
def finding_nearest_restaurant(nav, restaurant_data, customer_location):
    number_restaurants = 5
    travel_times = []
    for restaurant_num, restaurant_info in restaurant_data.items():
        restaurant_location = [restaurant_info['object_']['x'], restaurant_info['object_']['y']]
        customer_loc, restaurant_loc = nav.find_nodes([customer_location, restaurant_location])
        route = nav.find_routes(customer_loc, restaurant_loc, weight="time")
        time, distance = nav.find_route_lengths(route)
        travel_times.append([time, restaurant_num])
    nearest_restaurants = sorted(travel_times, key=lambda x: x[0])[0:number_restaurants]
    return nearest_restaurants

'''
This function computes and returns the euclidean distance between two coordinates.
'''
def calculate_distance(coordinate1, coordinate2):
    diff_x = abs(coordinate1[0] - coordinate2[0])
    diff_y = abs(coordinate1[1] - coordinate2[1])
    distance = math.sqrt(diff_x * 2 + diff_y * 2)
    return distance


'''
This function returns a list of list of nearest restaurants. This is based on the
Euclidean distance. This version use customer coordinate.
'''
def finding_nearest_restaurant2(restaurant_data, customer_location):
    number_restaurants = 5
    travel_distances = []
    for restaurant_num, restaurant_info in restaurant_data.items():
        restaurant_location = [restaurant_info['object_']['x'], restaurant_info['object_']['y']]
        distance = calculate_distance(restaurant_location, customer_location)
        travel_distances.append([distance, restaurant_num])
    nearest_restaurants = sorted(travel_distances, key=lambda x: x[0])[0:number_restaurants]
    return nearest_restaurants

'''
update_order_dict() updates the global order dictionary after every action such that every time the 
availability, ETA and duration (time until actionable again) is known.
'''
def update_order_dict(order_dict_without_time, order_dict_with_time, sim, time_gap, initialize):
    if initialize == True:
        for order_dict in sim.vehicle_orders.values():
            for order_num, order_info in order_dict.items():
                order_dict_with_time[order_num] = order_info
                order_dict_with_time[order_num]["time online"] = 0
                order_dict_with_time[order_num]["uncertainty"] = 0
    else:
        orders_read = []
        for order_dict in sim.vehicle_orders.values():
            for order_num, order_info in order_dict.items():
                if order_num not in orders_read:
                    orders_read.append(order_num)
                    if order_num in order_dict_with_time.keys():
                        order_dict_with_time[order_num]["time online"] += time_gap
                    else:
                        order_dict_with_time[order_num] = order_info
                        order_dict_with_time[order_num]["time online"] = time_gap
                        order_dict_with_time[order_num]["uncertainty"] = time_gap
        missing_orders = [key for key in order_dict_with_time if key not in orders_read]
        if len(missing_orders) > 0:
            for order_num in missing_orders:
                order_dict_with_time.pop(order_num)
    return order_dict_with_time

'''
update_vehicle_dict() updates the global vehicle_dict dictionary after every action such that every time the 
availability, ETA and duration (time until actionable again) is known.
'''
def update_vehicle_dict(sim, vehicle_dict, start_location, current_time, current_vehicle, nav):
    end_location = sim.vehicle_locations[sim.current_vehicle]
    end_location_node, start_location_node = nav.find_nodes([end_location, start_location])
    route = nav.find_routes(end_location_node, start_location_node)
    travel_time = nav.find_route_lengths(route)[0]
    total_distance = nav.find_route_lengths(route)[1]
    for vehicle in vehicle_dict:
        ETA = np.uint32(sim.vehicle_times[vehicle]).item()  # Force that time is an int variable
        tuple_tolist = list(vehicle_dict[vehicle])
        duration = tuple_tolist[3]

        if sim.current_time >= ETA:
            availability = "Available"
        else:
            availability = "Unavailable"

        if vehicle == current_vehicle:
            duration = ETA - sim.current_time
        elif duration != 0:
            duration = duration - (sim.current_time - current_time)
        else:
            duration = 0

        vehicle_dict[vehicle] = tuple(tuple_tolist)
        vehicle_dict[vehicle] = (availability, sim.vehicle_locations[vehicle],
                                 dt.timedelta(seconds=ETA), duration)
    return vehicle_dict

"""
    Initializes the vehicle dictionary at the beginning of the simulation.

    Parameters:
    - no_vehicles (int): The total number of vehicles in the simulation.
    - sim (Simulation): An instance of the simulation class.

    Returns:
    dict: The initialized vehicle dictionary with information about the availability and status of each vehicle.
"""
def initialize_dict(no_vehicles, sim):
    all_vehicle_ids = list(range(no_vehicles))
    available_vehicles_dict = {}
    for vehicle in range(no_vehicles):
        if sim.current_vehicle not in available_vehicles_dict.keys():
            available_vehicles_dict[sim.current_vehicle] = (
                "Available", sim.current_location, dt.timedelta(seconds=sim.current_time), 0)
            sim.wait(0)
        else:
            continue

    unavailable_vehicle_ids = list(
        set(available_vehicles_dict.keys()) ^ set(all_vehicle_ids))  # Obtain vehicles ids that are not available
    unavailable_vehicle_dict = {}
    for id in unavailable_vehicle_ids:
        unavailable_vehicle_dict[id] = ("Unavailable", sim.vehicle_locations[id], "Unstarted", 0)
    vehicle_dict = {**available_vehicles_dict, **unavailable_vehicle_dict}

    return vehicle_dict

    
'''
This function is used to sign area to a color based on the heatmap values.
'''
def heatmap_per_region(nav):
    heatmap = import_json_file("data/heatmap_v1.json")
    
    area_values = defaultdict(list)
    
    for node in heatmap.keys():
        coordinates = [nav.nodes[int(node)][0],nav.nodes[int(node)][1]]
        area = find_area(coordinates, nav)
        if area != None:
            area_values[area].append(heatmap[node])
    
    area_values2 = dict()
    for area, values in area_values.items():
        area_values2[area] = np.mean(values) 
        
    diff = max(area_values2.values())-min(area_values2.values())
    steps = diff/9
      
    bins = []
    for i in range(1,10):
        value = min(area_values2.values()) + steps*i
        if i == 9:
            bins.append(2)
        else:
            bins.append(value)
    
    # Dit stukje is een beetje omslachtig maar het zorgt ervoor dat dezelfde gebieden als in de heatmap de juiste kleur krijgen
    for area, value in area_values2.items():
        count = 1
        for max_limit in bins:
            if value < max_limit:
                if count <= 3:
                    area_values2[area] = "red"
                elif count <= 6:
                    area_values2[area] = "orange"
                else:
                    area_values2[area] = "green"
                break
            count += 1
            
    return area_values2

'''
This function read orders and check if this is a valid order. An order is
valid is the customer and restaurant location is in a green area. It will
return a dictionary of valid orders.
'''
def check_area_orders(sim_orders, area_heatmap, nav):
    valid_orders = dict()
    for order_num, order_info in sim_orders.items():
        customer_area = find_area(order_info["customer"], nav)
        restaurant_area = find_area(order_info["restaurant"], nav)
        if area_heatmap[customer_area] == "green" and area_heatmap[restaurant_area] == "green":
            valid_orders[order_num] = order_info
    return valid_orders




