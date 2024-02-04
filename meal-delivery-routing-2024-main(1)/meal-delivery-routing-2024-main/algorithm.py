from simulator import Simulator
from navigator import Navigator
from datetime import datetime
import support_functions as sf
import numpy as np
import time

'''
This function will return a dictionary of orders with the reward.
This is stage 1 of the algorithm. In this stage the reward is the fee
'''
def algorithm_1(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, no_vehicles, beta1, beta2, beta3,
                     vehicle_distances, day_of_week_string):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        orders_with_reward[order_number] = order_information['fee']
    return orders_with_reward

'''
This function will return a dictionary of orders with the reward.
This is stage 2 of the algorithm. In this stage the reward is the fee/(travel time)
'''
def algorithm_2(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, no_vehicles, beta1, beta2, beta3,
                     vehicle_distances, day_of_week_string):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        end_location_area = sf.find_area(order_information['customer'], nav)
        current_location_area = sf.find_area(sim.current_location, nav)
        pickup_area = sf.find_area(order_information['restaurant'], nav)
        if ((
                pickup_area == current_location_area == end_location_area) or pickup_area is None or current_location_area is None or end_location_area is None):
            start, pickup, finish = nav.find_nodes(
                [sim.current_location, order_information['restaurant'], order_information['customer']])
            routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
            times, distances = nav.find_route_lengths(routes)
            sum_time = sum(times)
        else:
            times2_list = []
            times2_list.append(vehicle_distances[current_location_area][pickup_area])
            times2_list.append(vehicle_distances[pickup_area][end_location_area])
            sum_time = sum(times2_list)
            finish = nav.find_nodes(order_information['customer'])

        orders_with_reward[order_number] = order_information['fee'] / sum_time
    return orders_with_reward

'''
This function returns a dictionary of orders with the
reward: (highest fee)/(travel time)+F(end_location).
Where F(end_location) = (1+X), where X is the percentage of total orders covered in 
the nearest restaurants from this end location. The nearest restaurants from the end location
are based on travel time. This is the first version of stage 3 of the algorithm.
'''
def algorithm_3_v1(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, vehicles, beta1, vehicle_distances):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        # fee/distance calculation
        start, pickup, finish = nav.find_nodes(
            [sim.current_location, order_information['restaurant'], order_information['customer']])
        routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
        times, distances = nav.find_route_lengths(routes)

        time = sum(times)
        reward_one = order_information['fee'] / time * 100

        # F(end_location) calculation
        nearest_restaurants = sf.finding_nearest_restaurant(nav, restaurant_data, order_information['customer'])
        percentage = 0
        for restaurant in nearest_restaurants:
            percentage += restaurant_data[restaurant[1]]['percentage_orders']
        reward_two = 1 + percentage
        orders_with_reward[order_number] = reward_one + reward_two
    return orders_with_reward


'''
This function returns a dictionary of orders with the 
reward: (highest fee)/(travel time)+F(end_location).
Where F(end_location) = (1+X), where X is the percentage of total orders covered in 
the nearest restaurants from this end location. The nearest restaurants from the end location
are based on Euclidean distance. This is the second version of stage 3 of the algorithm.
'''
def algorithm_3_v2(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, vehicles, beta1, vehicle_distances):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        # fee/distance calculation
        start, pickup, finish = nav.find_nodes(
            [sim.current_location, order_information['restaurant'], order_information['customer']])
        routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
        times, distances = nav.find_route_lengths(routes)
        time = sum(times)
        reward_one = order_information['fee'] / time * 100

        # F(end_location) calculation
        nearest_restaurants = sf.finding_nearest_restaurant2(restaurant_data, order_information['customer'])
        percentage = 0
        for restaurant in nearest_restaurants:
            percentage += restaurant_data[restaurant[1]]['percentage_orders']
        reward_two = 1 + percentage
        orders_with_reward[order_number] = reward_one + reward_two
    return orders_with_reward


'''
This function returns a dictionary that containts every available order with the
reward:(highest fee)/(travel time)+F(end_location).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)),
where X is the percentage of total orders covered from the nearest restaurants from this
end location. Y is the percentage of total_weighted_average_fee covered from the nearest
restaurants from this endlocation. The nearest restaurants from the end location
are based on a heatmap (see support_functions.py for calculations for this heatmap).
This is the third version of stage 3 of the algorithm.
'''
def algorithm_3_v3(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, vehicles, beta1, beta2, beta3,
                   vehicle_distances, day):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        end_location_area = sf.find_area(order_information['customer'], nav)
        current_location_area = sf.find_area(sim.current_location, nav)
        pickup_area = sf.find_area(order_information['restaurant'], nav)
        if ((
                pickup_area == current_location_area == end_location_area) or pickup_area is None or current_location_area is None or end_location_area is None):
            start, pickup, finish = nav.find_nodes(
                [sim.current_location, order_information['restaurant'], order_information['customer']])
            routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
            times, distances = nav.find_route_lengths(routes)
            sum_time = sum(times)
        else:
            times2_list = []
            times2_list.append(vehicle_distances[current_location_area][pickup_area])
            times2_list.append(vehicle_distances[pickup_area][end_location_area])
            sum_time = sum(times2_list)
            finish = nav.find_nodes(order_information['customer'])

        reward_one = (order_information['fee'] / sum_time * 100) * beta1

        # F(end_location) calculation
        reward_two = heatmap[str(finish)]
        orders_with_reward[order_number] = reward_one + reward_two
    return orders_with_reward

'''
This function returns a dictionary that containts every available order with the
reward:(highest fee)/(travel time)+F(end_location).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)),
where X is the percentage of total orders covered from the nearest restaurants from this
end location. Y is the percentage of total_weighted_average_fee covered from the nearest
restaurants from this endlocation. The nearest restaurants from the end location
are based on a heatmap (see support_functions.py for calculations for this heatmap).
This is the third version of stage 3 of the algorithm. This one is tweaked in order to use heatmap 2.
'''
def algorithm_3_heatmap2(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, vehicles, beta1, beta2, beta3,
                   vehicle_distances, day):
    orders_with_reward = dict()
    for order_number, order_information in dict_orders.items():
        end_location_area = sf.find_area(order_information['customer'], nav)
        current_location_area = sf.find_area(sim.current_location, nav)
        pickup_area = sf.find_area(order_information['restaurant'], nav)
        if ((
                pickup_area == current_location_area == end_location_area) or pickup_area is None or current_location_area is None or end_location_area is None):
            start, pickup, finish = nav.find_nodes(
                [sim.current_location, order_information['restaurant'], order_information['customer']])
            routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
            times, distances = nav.find_route_lengths(routes)
            sum_time = sum(times)
        else:
            times2_list = []
            times2_list.append(vehicle_distances[current_location_area][pickup_area])
            times2_list.append(vehicle_distances[pickup_area][end_location_area])
            sum_time = sum(times2_list)
            finish = nav.find_nodes(order_information['customer'])

        reward_one = (order_information['fee'] / sum_time * 100) * beta1

        # F(end_location) calculation
        reward_two = heatmap[day][str(finish)]
        orders_with_reward[order_number] = reward_one + reward_two
    return orders_with_reward

'''
This function returns a dictionary that containts every available order with the
reward:(highest fee)/(travel time)+F(end_location).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)),
where X is the percentage of total orders covered from the nearest restaurants from this
end location. Y is the percentage of total_weighted_average_fee covered from the nearest
restaurants from this endlocation. The nearest restaurants from the end location
are based on a heatmap (see support_functions.py for calculations for this heatmap).
This is the third version of stage 3 of the algorithm. Tweaked so we can use it in the dashboard.
'''
def algorithm_3_dash(dict_vehicles, nav, sim, restaurant_data, heatmap, vehicle_dict, no_vehicles, beta1, beta2, beta3,
                     vehicle_distances, day_of_week_string):
    vehicles_with_orders_and_rewards = dict()
    for vehicle_number, vehicle_orders in dict_vehicles.items():
        orders_with_reward = dict()
        for order_number, order_information in vehicle_orders.items():
            end_location_area = sf.find_area(order_information['customer'], nav)
            current_location_area = sf.find_area(vehicle_dict[vehicle_number][1], nav)
            pickup_area = sf.find_area(order_information['restaurant'], nav)
            if ((pickup_area == current_location_area == end_location_area)
                    or pickup_area is None or current_location_area is None or end_location_area is None):
                start, pickup, finish = nav.find_nodes([vehicle_dict[vehicle_number][1], order_information['restaurant'], order_information['customer']])
                routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
                times, distances = nav.find_route_lengths(routes)
                sum_time = sum(times)
            else:
                times2_list = []
                times2_list.append(vehicle_distances[current_location_area][pickup_area])
                times2_list.append(vehicle_distances[pickup_area][end_location_area])
                sum_time = sum(times2_list)
                finish = nav.find_nodes(order_information['customer'])

            reward_one = beta1 * (order_information['fee'] / sum_time * 100)
            reward_two = beta2 * heatmap[str(finish)]
            orders_with_reward[order_number] = reward_one + reward_two

        vehicles_with_orders_and_rewards[vehicle_number] = orders_with_reward

    return vehicles_with_orders_and_rewards

'''
This function returns a dictionary that containts every available order with the
reward:(highest fee)/(travel time)+F(end_location) + G(nend, tend).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)),
G(nend, tend)= (1-11+min(vehicle_times)) *(1-11+min(delta_times) 
where X is the percentage of total orders covered from the nearest restaurants from this
end location. Y is the percentage of total_weighted_average_fee covered from the nearest
restaurants from this endlocation. The nearest restaurants from the end location
are based on a heatmap (see support_functions.py for calculations for this heatmap).
This is the fourth version of the algorithm. Tweaked in order to be used in the dashboard.
'''
def algorithm_4_dash(dict_vehicles, nav, sim, restaurant_data, heatmap, vehicle_dict, no_vehicles, beta1, beta2, beta3,
                     vehicle_distances, day_of_week_string):

    vehicles_with_orders_and_rewards = dict()
    for vehicle_number, vehicle_orders in dict_vehicles.items():
        orders_with_reward = dict()
        for order_number, order_information in vehicle_orders.items():
            end_location_area = sf.find_area(order_information['customer'], nav)
            current_location_area = sf.find_area(vehicle_dict[vehicle_number][1], nav)
            pickup_area = sf.find_area(order_information['restaurant'], nav)

            if ((pickup_area == current_location_area == end_location_area) or (pickup_area is None)
                    or (current_location_area is None) or (end_location_area is None)):
                start, pickup, finish = nav.find_nodes([vehicle_dict[vehicle_number][1],
                                        order_information['restaurant'], order_information['customer']])
                routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
                times_list, distances = nav.find_route_lengths(routes)
                sum_time = sum(times_list)
            else:
                times_list = []
                times_list.append(vehicle_distances[current_location_area][pickup_area])
                times_list.append(vehicle_distances[pickup_area][end_location_area])
                sum_time = sum(times_list)
                finish = nav.find_nodes(order_information['customer'])

            g_list = []
            for id in vehicle_dict:
                if id != vehicle_number:
                    tuple_tolist = list(vehicle_dict[id])
                    duration = tuple_tolist[3]
                    vehicle_area = sf.find_area(sim.vehicle_locations[id], nav)
                    if (vehicle_area is None) or (current_location_area is None):
                        start_node, end_node = nav.find_nodes([vehicle_dict[vehicle_number][1], sim.vehicle_locations[id]])
                        route = nav.find_routes(start_node, end_node, weight="time")
                        time = nav.find_route_lengths(route)[0]
                        g_list.append(np.abs(sum_time - duration) * time)
                    else:
                        time = vehicle_distances[current_location_area][vehicle_area]
                        g_list.append(np.abs(sum_time - duration) * time)
            alpha1 = 0.075
            G_multiplier = (1 - 1 / (1 + alpha1 * int(min(g_list))))

            if sum_time != 0:
                reward_one = beta1 * (order_information['fee'] * 100 / sum_time)
            else:
                reward_one = 0
            reward_two = beta2 * heatmap[str(finish)] + beta3 * G_multiplier
            orders_with_reward[order_number] = reward_one + reward_two
        vehicles_with_orders_and_rewards[vehicle_number] = orders_with_reward

    return vehicles_with_orders_and_rewards

'''
This function returns a dictionary that contains every available order with the
reward:(highest fee)/(travel time)+F(end_location) + G(nend, tend).
Where F(end_location) = (1+X)*(1+Y)*(1/(1+total_travel_time_nearest_restaurants/10.000)),
G(nend, tend)= (1-11+min(vehicle_times)) *(1-11+min(delta_times) 
where X is the percentage of total orders covered from the nearest restaurants from this
end location. Y is the percentage of total_weighted_average_fee covered from the nearest
restaurants from this endlocation. The nearest restaurants from the end location
are based on a heatmap (see support_functions.py for calculations for this heatmap).
This is the fourth version of the algorithm.
'''
def algorithm_4(dict_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, no_vehicles, beta1, beta2, beta3,
                vehicle_distances, dayofweek_string):
    orders_with_reward = dict()
    other_vehicle_nodes = np.delete(nav.find_nodes(sim.vehicle_locations), sim.current_vehicle)

    for order_number, order_information in dict_orders.items():
        # Fee/distance calculation
        end_location_area = sf.find_area(order_information['customer'], nav)
        current_location_area = sf.find_area(sim.current_location, nav)
        pickup_area = sf.find_area(order_information['restaurant'], nav)
        if ((pickup_area == current_location_area == end_location_area) or pickup_area is None
                or current_location_area is None or end_location_area is None):
            start, pickup, finish = nav.find_nodes(
                [sim.current_location, order_information['restaurant'], order_information['customer']])
            routes = nav.find_routes([start, pickup], [pickup, finish], weight="time").diagonal()
            times2_list, distances = nav.find_route_lengths(routes)
            sum_time = sum(times2_list)
        else:
            times2_list = []
            times2_list.append(vehicle_distances[current_location_area][pickup_area])
            times2_list.append(vehicle_distances[pickup_area][end_location_area])
            sum_time = sum(times2_list)
            finish = nav.find_nodes(order_information['customer'])

        delta_list = []
        for id in vehicle_dict:
            tuple_tolist = list(vehicle_dict[id])
            duration = tuple_tolist[3]
            delta_list.append(np.abs(sum_time - duration))

        alpha1 = 0.075
        alpha2 = 0.1

        locations_multiplier = (1 - 1 / (1 + alpha1 * int(min(times2_list)))) + 0.5
        delta_multiplier = (1 - 1 / (1 + alpha2 * int(min(delta_list)))) + 0.5

        # reward_one_og = (order_information['fee'] * 100 / sum_time)
        reward_one = (order_information['fee'] * 100 / sum_time) * beta1

        # reward_F = heatmap[str(finish)]
        # reward_F_b = reward_F*beta2

        # reward_G = locations_multiplier * delta_multiplier
        # reward_G_b = reward_G*beta3
        reward_two = beta2 * heatmap[str(finish)] + beta3 * locations_multiplier * delta_multiplier

        orders_with_reward[order_number] = reward_one + reward_two
        #print(f'reward_one og: {reward_one_og}, reward F: {reward_F}, reward G: {reward_G}')
        #print(f'reward one beta: {reward_one}, reward F beta: {reward_F_b}, reward G beta: {reward_G_b}')
    return orders_with_reward

'''
Runs the meal delivery simulation for a specified date range using the given algorithm.
returns The total sum of profits over the specified date range.
'''
def runsimulator(start_date, end_date, month, vehicles, algorithm, nav, restaurant_data, heatmap, profit_dict,
                 distance_dict, time_dict, beta1, beta2, beta3, vehicle_distances):
    temp_dict = []
    for day in range(start_date, end_date):
        print("-" * 75)
        print(f'Month:{month}')
        print(f'Day:{day}')
        print("")
        print("simulating...")
        sim = Simulator(2023, month, day)
        sim.reset(n_vehicles=vehicles)
        date_object = datetime(2023, month, day)
        day_of_week_string = date_object.strftime('%A')
        profit_count = 0
        fee_count = 0
        distance_count = 0
        start_time = time.time()
        vehicle_dict = sf.initialize_dict(vehicles, sim)
        while (sim.finished != True):
            if sim.orders:
                current_vehicle = sim.current_vehicle
                current_time = sim.current_time
                current_orders = sim.orders
                start_location = sim.current_location.copy()
                vehicle_dict = sf.update_vehicle_dict(sim, vehicle_dict, start_location, current_time, current_vehicle, nav)
                order_num = algorithm(sim.vehicle_orders, nav, sim, restaurant_data, heatmap, vehicle_dict, vehicles, beta1,
                                      beta2, beta3, vehicle_distances, day_of_week_string)
                order_highest_reward = max(order_num[current_vehicle], key=order_num[current_vehicle].get)
                profit, distance = sim.pickup(order=order_highest_reward)
                fee = current_orders[order_highest_reward]["fee"]
                profit_count = profit_count + profit
                fee_count = fee_count + fee
                distance_count = distance_count + distance
            else:
                sim.wait(300)
        end_time = time.time()
        elapsed_time = end_time - start_time
        profit_dict[month, day] = fee_count
        distance_dict[month, day] = distance_count
        time_dict[month, day] = elapsed_time

        print(f'profit: {profit_count}')
        print(f'fee {fee_count}')
        print(f'distance travelled: {distance_count}')
        print("-" * 75)

        temp_dict.append(fee_count)
        temp_dict_sum = sum(temp_dict)

    return temp_dict_sum
'''
Runs the meal delivery simulation for a specified date range using the given algorithm. In order to analyze the 
duration of each order, how long does it take before it disappears? 
'''
def runsimulator_order_duration(start_date, end_date, month, vehicles, algorithm, nav, restaurant_data, heatmap, profit_dict,
                 distance_dict, time_dict, beta1, beta2, beta3, vehicle_distances):
    for day in range(start_date, end_date):
        count = 1
        print("-"*75)
        print(f'Month:{month}')
        print(f'Day:{day}')
        print("")
        print("simulating...")
        sim = Simulator(2023, month, day)
        sim.reset(n_vehicles=vehicles)
        profit_count = 0
        distance_count = 0 
        # start_time = time.time()
        date_object = datetime(2023, month, day)
        time_list = []
        order_dict_with_time = dict()
        time_gap = 0
        order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, time_gap, True)
        while(sim.finished!=True):
            if sim.orders:
                s1 = sim.current_time
                order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, time_gap, False)
                print(order_dict_with_time)
                order_with_rewards = algorithm(sim.orders, nav, sim, restaurant_data, heatmap, date_object.strftime('%A'), time_gap, beta1=8, beta2=4, beta3=8, vehicle_distances=vehicle_distances, day=day)
                order_highest_reward = max(order_with_rewards, key = order_with_rewards.get)
                order_dict_with_time.pop(order_highest_reward)
                profit, distance = sim.pickup(order=order_highest_reward)
            else:
                s1 = sim.current_time
                order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, time_gap, False)
                sim.wait(300)
            e1 = sim.current_time
            time_gap = e1-s1
            print(time_gap)
            print("-"*70)
            count += 1
            if count == 6:
                break
        print(f'profit: {profit_count}')
        print(f'distance travelled: {distance_count}')
        print("-"*75)


if __name__ == "__main__":
    nav = Navigator("data/paris_map.txt")
    heatmap_v1 = sf.import_json_file("data/heatmap_v1.json")
    heatmap_v2 = sf.import_json_file("data/heatmap_v2.json")
    restaurant_data = sf.import_json_file("data/restaurant_data_v2.json")

    weeks = sf.import_json_file("data/weeks_used_for_performance_measure.json")
    vehicle_distances = sf.import_pickle_file('data/area_distances_dict.pkl')

    num_vehicles = 10
    profit_dict = dict()
    distance_dict = dict()
    time_dict = dict()

    for month in range(1, 13):
        runsimulator(weeks[str(month)][0], weeks[str(month)][1], month, num_vehicles, algorithm_4_dash, nav, restaurant_data,
                     heatmap_v1, profit_dict, distance_dict, time_dict, 8, 2, 0, vehicle_distances)

    sf.export_pickle_file(profit_dict, "data/data_algoritme4/performance_alg4_profit_heatmap1_10.pickle")
    sf.export_pickle_file(distance_dict, "data/data_algoritme4/performance_alg4_distance_heatmap1_10.pickle")
    sf.export_pickle_file(time_dict, "data/data_algoritme4/performance_alg4_runtime_heatmap1_10.pickle")




