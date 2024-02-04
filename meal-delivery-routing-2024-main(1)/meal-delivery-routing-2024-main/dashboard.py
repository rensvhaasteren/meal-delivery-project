import sys
from dash import Dash, html, dcc, dash_table, Input, Output, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from simulator import Simulator
from navigator import Navigator
from datetime import date
import datetime as dt
import numpy as np
import pandas as pd
import support_functions as sf
import algorithm
from PIL import Image
import base64
from io import BytesIO
from cartopy.io.img_tiles import OSM
import dash_daq as daq
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

'''
Prematurely define the dataframe table headers
'''
vehicle_columns = ["ID", "Availability", "Location", "ETA", "Duration"]
empty_vehicle_df = pd.DataFrame(columns=vehicle_columns)

order_columns = ["ID", "Pick-up time", "Delivery time", "Age range", "Fee"]
empty_order_df = pd.DataFrame(columns=order_columns)

combination_columns = ["Vehicle", "Order", "Reward"]
empty_combination_df = pd.DataFrame(columns=combination_columns)

'''
Dashboard layout
'''
app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Card([
                        dbc.CardBody([
                            html.H1("Meal Delivery Routing in Paris",
                                    style={'textAlign': 'left', 'color': '#198754', 'font-weight': 'bold'}),
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5("Time:"),
                                        width = 7),
                                        dbc.Col(
                                            html.Div(id="time", style={'color': '#198754', 'fontSize': 20, "font-weight": "bold"}),
                                        width = 5),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5("Actionable vehicle:"),
                                        width = 9),
                                        dbc.Col(
                                            html.Div(id="chosen_vehicle", style={'color': '#198754', 'fontSize': 20, "font-weight": "bold"}),
                                        width = 3)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5("Order velocity:"),
                                        width = 9),
                                        dbc.Col(
                                            html.Div(id="order_velocity", style={'color': '#198754', 'fontSize': 20, "font-weight": "bold"}),
                                        width = 3)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5("Min. order age, 1% error (min):"),
                                        width = 9),
                                        dbc.Col(
                                            html.Div(id="quantile", style={'color': '#198754', 'fontSize': 20, "font-weight": "bold"}),
                                        width = 3)
                                    ]),
                                ], width = 5),
                                dbc.Col([
                                    dbc.Row([
                                        dcc.Dropdown(id="order-dropdown", placeholder="Select an order", style={'width': '300px'})
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Input(id="wait_input", type="number", placeholder="Wait duration (min)", style={'width': '300px'}),
                                        width = 10),
                                        dbc.Col(
                                            dbc.Button(id='wait_submit', type='submit', children='ok', outline=True, color="success", className="me-1"),
                                        width = 1)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Input(id="relocate_x_input", type="number", placeholder="Longitude (N)", style={"width": "150px"}),
                                        width = 5),
                                        dbc.Col(
                                            dbc.Input(id="relocate_y_input", type="number", placeholder="Latitude (E)", style={"width": "150px"}),
                                        width = 5),
                                        dbc.Col(
                                            dbc.Button(id='relocate_submit', type='submit', children='ok', outline=True, color="success", className="me-1"),
                                        width = 1)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                                dbc.Button(children="Relocate to nearest restaurant", type='submit', id="relocate_to_restaurant", n_clicks=0, outline=True, color="success", className="me-1"),
                                        width=12)
                                    ]),
                                ])
                            ]),
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H6("Date"),
                                    dcc.DatePickerSingle(
                                        id='date',
                                        min_date_allowed=date(date.today().year - 10, 1, 1),
                                        max_date_allowed=date(date.today().year + 10, 12, 31),
                                        initial_visible_month=date(date.today().year, date.today().month,
                                                                   date.today().day),
                                        date=date(date.today().year, date.today().month, date.today().day),
                                        with_portal=False,
                                        first_day_of_week=2
                                    )
                                ])
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.H6("Number of vehicles"),
                                    dbc.Input(id="no_vehicles", type="number", placeholder="Select at start")
                                ]),
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.H6("Initialize"),
                                    dbc.Button(children="Start", type='submit', id="initialize_input", n_clicks=0, outline=True, color="success", className="me-1"),
                                ]),
                            ], width=2),
                            dbc.Col([
                                html.H6("Map off | Map on"),
                                daq.BooleanSwitch(
                                    id='toggle_map',
                                    on=False,
                                    color="#198754",
                                    labelPosition="top"
                                ),
                            ], width=3)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H6("Cumulative revenue"),
                                    html.Div(id="revenue_block", style={'border': '1px solid #198754', 'padding': '10px'}),
                                ]),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.H6("Total number of orders"),
                                    html.Div(id="num_orders_block", style={'border': '1px solid #198754', 'padding': '10px'}),
                                ]),
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H6("Distance travelled"),
                                    html.Div(id="km_travelled_block", style={'border': '1px solid #198754', 'padding': '10px'}),
                                ]),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.H6("Income velocity"),
                                    html.Div(id="income_velocity", style={'border': '1px solid #198754', 'padding': '10px'}),
                                ]),
                            ], width=6),
                        ]),
                    ]),
                ])
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Vehicle Information:", className="card-title",
                                style={'color': '#198754', 'font-weight': 'bold', 'font-size': '1.5em'}),
                        html.Div(id="vehicle-info-output"),
                        dash_table.DataTable(
                            empty_vehicle_df.to_dict('records'),
                            columns=[{"name": col, "id": col} for col in vehicle_columns],
                            id='vehicle_table'
                        )
                    ], style={'backgroundColor': '#ededed', 'border': '2px solid black'})
                ])
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Order Information:", className="card-title",
                                style={'color': '#198754', 'font-weight': 'bold', 'font-size': '1.5em'}),
                        html.Div(id="order-info-output"),
                        dash_table.DataTable(
                            empty_order_df.to_dict('records'),
                            columns=[{"name": col, "id": col} for col in order_columns],
                            id='order_table',
                            fixed_rows={'headers': True}
                        )
                    ], style={'backgroundColor': '#ededed', 'border': '2px solid black'})
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Order - Vehicle Combinations:", className="card-title",
                                style={'color': '#198754', 'font-weight': 'bold', 'font-size': '1.5em'}),
                        html.Div(id="combination-output"),
                        dash_table.DataTable(
                            empty_combination_df.to_dict('records'),
                            columns=[{"name": col, "id": col} for col in combination_columns],
                            id='combi_table',
                            fixed_rows={'headers': True},
                        )
                    ], style={'backgroundColor': '#ededed', 'border': '2px solid black'})
                ])
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Img(id='map')
            ], width=20),
            html.Img(id='legend')
        ]),
    ]),
])

'''
Globals
'''
vehicle_startlocations = {}
vehicle_dict = {}
restaurant_locations = {}
order_dict_with_time = {}
total_revenue = 0
no_orders = 0
total_tip = 0
total_distance = 0
no_vehicles = 0
start_of_day_time = 0
date_selected = date.today()
sim = Simulator(date.today().year, date.today().month, date.today().day)

'''
Functions
'''

'''
The get_order_velocity_and_quantile() function looks through the order_velocity json file for
a matching number of vehicles, name of the weekday of the selected date and current hour + it returns the 1%-quantile of
the weekday regarding the order age
'''
def get_order_velocity_and_quantile():
    time = dt.timedelta(seconds=sim.current_time)
    hour = time.seconds // 3600
    weekday = date_selected.weekday()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    name_weekday = days[weekday]
    quantile = quantile_weekday[name_weekday]/60
    return order_velocity_data[str(no_vehicles)][name_weekday][str(hour + 1)], quantile

'''
The get_orders() function displays the orders available for the currently actionable vehicle,
notice the age - uncertainty as lowerbound since we can't know for sure when an order arrives on the market
'''
def get_orders(orders, order_dict_with_time):
    order_info_list = []
    for key, value in orders.items():
        restaurant_node, customer_node = nav.find_nodes([value['restaurant'], value['customer']])
        route = nav.find_routes(restaurant_node, customer_node)
        time, distance = nav.find_route_lengths(route)
        age = order_dict_with_time[key]["time online"]
        uncertainty = order_dict_with_time[key]["uncertainty"]
        info_dict = {
            "ID": key,
            "Pick-up time": str(dt.timedelta(seconds=value['pickup_time'].item())),
            "Delivery time": str(dt.timedelta(seconds=time)),
            "Age range": str(round((age - uncertainty)/60, 2)) + " - " + str(round(age/60, 2)) + " min",
            "Fee": str("€ "+str(round(value['fee'], 2)))
        }

        order_info_list.append(info_dict)
    if len(order_info_list) == 0:
        order_df = pd.DataFrame(order_info_list, columns=order_columns)
    else:
        order_df = pd.DataFrame(order_info_list, columns=order_columns).sort_values(by='Fee', ascending=False)

    return order_df

'''
The get_vehicle_info() function acts in a similar matter as the previous function, it return the format to be able 
to display the information of all vehicles, notice that the location of unavailable vehicles is the end location of
that vehicle instead of the actual location
'''
def get_vehicle_info(vehicle_dict):
    vehicle_info_list = []
    if vehicle_dict is not None:
        for vehicle in vehicle_dict:
            availability = vehicle_dict[vehicle][0]
            location = vehicle_dict[vehicle][1]
            end_time = vehicle_dict[vehicle][2]
            duration = vehicle_dict[vehicle][3]

            info_dict = {
                "ID": vehicle,
                "Availability": availability,
                "Location": str(location),
                "ETA": str(dt.timedelta(seconds = int(end_time.total_seconds()))),
                "Duration": str(dt.timedelta(seconds=duration))
            }
            vehicle_info_list.append(info_dict)

    vehicle_df = pd.DataFrame(vehicle_info_list, columns=vehicle_columns)
    return vehicle_df

'''
vehicle_order_combination() displays the vehicle order combinations and rewards in the order-vehicle combination table 
'''
def vehicle_order_combination(vehicles_with_orders_and_rewards):
    combination_info_list = []
    for vehicle, orders_with_reward in vehicles_with_orders_and_rewards.items():
        for order_id, reward in orders_with_reward.items():
            info_dict = {
                "Vehicle": vehicle,
                "Order": order_id,
                "Reward": round(reward, 3)
            }
            combination_info_list.append(info_dict)

    combination_df = pd.DataFrame(combination_info_list)
    if combination_info_list != []:
        combination_df = combination_df.sort_values(by='Reward', ascending=False)
    return combination_df

'''
update_vehicle_dict() updates the global vehicle_dict dictionary after every action such that every time the 
availability, ETA and duration (time until actionable again) is known.
'''
def update_vehicle_dict(vehicle_dict, start_location, current_time, current_vehicle):
    global vehicle_df
    end_location = sim.vehicle_locations[sim.current_vehicle]
    end_location_node, start_location_node = nav.find_nodes([end_location, start_location])
    for vehicle in vehicle_dict:
        ETA = np.uint32(sim.vehicle_times[vehicle]).item()  # Force that time is an int variable
        tuple_tolist = list(vehicle_dict[vehicle])
        duration = tuple_tolist[3]
        if ETA <= sim.current_time:
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
        vehicle_dict[vehicle] = (availability, sim.vehicle_locations[vehicle], dt.timedelta(seconds=ETA), duration)
    return vehicle_dict

'''
find_nearest_restaurant() is a function that determines the restaurant closest to the location of the current vehicle,
this calculation is based on area due to the execution time. Note that if the distance between the area with the 
current vehicle and the area of the restaurants is the least for more than 1 restaurant, it takes the first restaurant 
it looped over as closest.
'''
def find_nearest_restaurant(start_location):
    loc_nearest_res = [0, 0]
    time_to_nearest_restaurant = 10000
    vehicle_distances = sf.import_pickle_file("data/area_distances_dict.pkl")

    start_area = sf.find_area(sim.current_location, nav)
    nearest_res_id = 0
    for id, info in restaurant_data.items():
        res_loc = [info["object_"]["x"], info["object_"]["y"]]
        res_area = sf.find_area(res_loc, nav)
        time = vehicle_distances[start_area][res_area]
        if time < time_to_nearest_restaurant:
            time_to_nearest_restaurant = time
            loc_nearest_res = res_loc
            nearest_res_id = id

    nearest_res_loc = [restaurant_data[nearest_res_id]["object_"]["x"], restaurant_data[nearest_res_id]["object_"]["y"]]

    start_node = nav.find_nodes(sim.current_location)
    res_node = nav.find_nodes(nearest_res_loc)
    route = nav.find_routes(start_node, res_node, weight="time")
    distance_to_nearest_res = nav.find_route_lengths(route)[1]
    return loc_nearest_res, distance_to_nearest_res

'''
The update_map() function gets called when the toggle button is on, and will show the map with all the information in 
it. The plt figure needs to be encoded into html because Dash and matplotlib do not work well together, due to this 
the functionality to interact with the graph is lost.
'''
def update_map(vehicle_startlocations, restaurant_locations):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=OSM().crs))
    ax = nav.display_map(ax=ax)
    ax = nav.display_network(ax=ax)
    for id, location in enumerate(sim.vehicle_locations):
        node_start = nav.find_nodes(vehicle_startlocations[id])
        node_end = nav.find_nodes(location)
        ax = nav.display_nodes(node_start, ax=ax, node_color="black", node_shape="+", node_size=45)
        if (node_start == node_end) and (id not in restaurant_locations):
            continue
        else:
            ax = nav.display_nodes(node_end, ax=ax, node_color="red", node_shape="*", node_size=45)
            if id in restaurant_locations:
                ax = nav.display_nodes(nav.find_nodes(restaurant_locations[id]), ax=ax, node_color="green",
                                       node_shape="+", node_size=45)
                route1 = nav.find_routes(node_start, nav.find_nodes(restaurant_locations[id]))
                ax = nav.display_route(route1, ax=ax, edge_color="blue")
                route2 = nav.find_routes(nav.find_nodes(restaurant_locations[id]), node_end)
                ax = nav.display_route(route2, ax=ax, edge_color="blue")
            else:
                route = nav.find_routes(node_start, node_end)
                ax = nav.display_route(route, ax=ax, edge_color= "blue")

    if sim.orders is not None:
        for order in sim.orders:
            customer_node = nav.find_nodes(sim.orders[order].get("customer"))
            restaurant_node = nav.find_nodes(sim.orders[order].get("restaurant"))
            if customer_node != restaurant_node:
                ax = nav.display_nodes(restaurant_node, ax=ax, node_color="green",
                                       node_shape="+", node_size=45)
                ax = nav.display_nodes(customer_node, ax=ax, node_color="purple",
                                       node_shape="*", node_size=45)
                route = nav.find_routes(restaurant_node, customer_node)
                ax = nav.display_route(route, ax=ax, edge_color="yellow", width = 1)


    legend = Image.open("data\Legenda.jpg")
    legend = legend.resize((175, 105))
    height = legend.size[1]
    legend = np.array(legend).astype(float) / 255
    fig.figimage(legend, 0, fig.bbox.ymax - height)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii") # Embed the result in the html output.
    return f'data:image/png;base64,{fig_data}'

'''
This function is a conglomerate function that gets called after every action to update all necessary parts. 
'''
def update_everything(vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time):
    order_info_pd = get_orders(sim.orders, order_dict_with_time)
    vehicle_info_pd = get_vehicle_info(vehicle_dict)
    reward_info_pd = vehicle_order_combination(vehicles_with_orders_and_rewards)

    if sim.current_vehicle in restaurant_locations:
        del restaurant_locations[sim.current_vehicle]

    if on:
        map_html = update_map(vehicle_startlocations, restaurant_locations)
    else:
        map_html = ""

    return order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time

'''
Callbacks. The dashboard uses 6 callbacks, to pick an order, to wait, to relocate to given x and y, to relocate to 
nearest  restaurant, to intialize, and to toggle the map. This number could most likely be reduced by adding a switch 
case for each action, reducing the lines of code drastically, however due to time constraints, this is not done. 
'''

'''
callback def_pick activates when an order is chosen. This function calls sim.pickup.
'''
@app.callback([Output("revenue_block", "children"),
               Output("vehicle_table", "data", allow_duplicate=True),
               Output("time", "children", allow_duplicate=True),
               Output("order-dropdown", "options", allow_duplicate=True),
               Output("num_orders_block", "children", allow_duplicate=True),
               Output("km_travelled_block", "children", allow_duplicate=True),
               Output('combi_table', 'data', allow_duplicate=True),
               Output('order_table', 'data', allow_duplicate=True),
               Output("map", "src", allow_duplicate=True),
               Output("chosen_vehicle", "children", allow_duplicate=True),
               Output("income_velocity", "children", allow_duplicate=True),
               Output("order_velocity", "children", allow_duplicate=True),
               Output("quantile", "children", allow_duplicate=True)],
              [Input("order-dropdown", "value")],
              [State("toggle_map", "on")],
              prevent_initial_call=True)
def def_pick(value, on):
    global vehicle_dict, total_tip, total_revenue, no_orders, total_distance, vehicle_startlocations
    global no_vehicles, start_of_day_time, restaurant_locations, order_dict_with_time, date_selected
    if value is None:
        raise PreventUpdate

    current_vehicle = sim.current_vehicle
    current_time = sim.current_time
    current_orders = sim.orders
    start_location = sim.current_location.copy()
    vehicle_startlocations[current_vehicle] = start_location
    vehicle_startlocations = vehicle_startlocations.copy()
    restaurant_locations[current_vehicle] = current_orders[value]["restaurant"]

    profit, distance = sim.pickup(order = value)

    fee = current_orders[value]["fee"]
    tip = profit - fee
    total_tip += tip
    total_revenue += fee
    total_distance += distance
    no_orders += 1

    order_velocity, quantile = get_order_velocity_and_quantile()
    vehicle_startlocations[sim.current_vehicle] = sim.current_location
    vehicle_dict = update_vehicle_dict(vehicle_dict, start_location, current_time, current_vehicle)
    vehicles_with_orders_and_rewards = algorithm.algorithm_4_dash(sim.vehicle_orders, nav, sim, restaurant_data,
                                                                  heatmap, vehicle_dict, no_vehicles, 8,
                                                                  2, 0, vehicle_distances, "")
    order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, sim.current_time - current_time, False)

    order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time = update_everything(
        vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time)

    income_velocity = total_revenue*60*60/(no_vehicles*(sim.current_time - start_of_day_time))

    return (f'€{total_revenue:.2f}',
            vehicle_info_pd.to_dict('records'),
            str(dt.timedelta(seconds=sim.current_time)),
            list(sim.orders.keys()),
            no_orders,
            f'{total_distance/1000:.2f} km',
            reward_info_pd.to_dict('records'),
            order_info_pd.to_dict('records'),
            map_html,
            sim.current_vehicle,
            f'€{income_velocity:.2f}/hour/vehicle',
            f'{order_velocity:.1f}',
            f'{quantile:.1f}')

'''
callback def_wait activates when a time is chosen and 'ok' is hit in the dashboard. This function calls sim.wait.
'''
@app.callback([Output("vehicle_table", "data", allow_duplicate=True),
               Output("time", "children", allow_duplicate=True),
               Output("order-dropdown", "options", allow_duplicate=True),
               Output("wait_input", "value"),
               Output('combi_table', 'data', allow_duplicate=True),
               Output('order_table', 'data', allow_duplicate=True),
               Output("map", "src", allow_duplicate=True),
               Output("chosen_vehicle", "children", allow_duplicate=True),
               Output("income_velocity", "children", allow_duplicate=True),
               Output("order_velocity", "children", allow_duplicate=True),
               Output("quantile", "children", allow_duplicate=True)],
              [Input('wait_submit', 'n_clicks')],
              [State("wait_input", "value"),
               State("toggle_map", "on")],
              prevent_initial_call=True)
def def_wait(n_clicks, value, on):
    global vehicle_dict, vehicle_startlocations, no_vehicles, restaurant_locations, start_of_day_time
    global total_revenue, order_dict_with_time, date_selected
    if n_clicks is None:
        raise PreventUpdate

    current_vehicle = sim.current_vehicle
    current_time = sim.current_time
    start_location = sim.current_location.copy()
    vehicle_startlocations = vehicle_startlocations.copy()

    sim.wait(value*60)

    order_velocity, quantile = get_order_velocity_and_quantile()
    vehicle_startlocations[sim.current_vehicle] = sim.current_location
    vehicle_dict = update_vehicle_dict(vehicle_dict, start_location, current_time, current_vehicle)
    vehicles_with_orders_and_rewards = algorithm.algorithm_4_dash(sim.vehicle_orders, nav, sim, restaurant_data,
                                                                  heatmap, vehicle_dict, no_vehicles, 8,
                                                                  2, 0, vehicle_distances, "")
    order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, sim.current_time - current_time, False)

    order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time = update_everything(
        vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time)

    income_velocity = total_revenue*60*60/(no_vehicles*(sim.current_time - start_of_day_time))

    return [vehicle_info_pd.to_dict('records'),
            str(dt.timedelta(seconds=sim.current_time)),
            list(sim.orders.keys()),
            "",
            reward_info_pd.to_dict('records'),
            order_info_pd.to_dict('records'),
            map_html,
            sim.current_vehicle,
            f'€{income_velocity:.2f}/hour/vehicle',
            f'{order_velocity:.1f}',
            f'{quantile:.1f}']

'''
callback def_relocate activates when a x and y coordinate is given and 'ok' is clicked. This function calls sim.relocate.
'''
@app.callback([Output("vehicle_table", "data", allow_duplicate=True),
               Output("time", "children", allow_duplicate=True),
               Output("order-dropdown", "options", allow_duplicate=True),
               Output("km_travelled_block", "children", allow_duplicate=True),
               Output('combi_table', 'data', allow_duplicate=True),
               Output('order_table', 'data', allow_duplicate=True),
               Output("map", "src", allow_duplicate=True),
               Output("chosen_vehicle", "children", allow_duplicate=True),
               Output("income_velocity", "children", allow_duplicate=True),
               Output("order_velocity", "children", allow_duplicate=True),
               Output("quantile", "children", allow_duplicate=True)],
              [Input("relocate_submit", "n_clicks")],
              [State("relocate_x_input", "value"),
               State("relocate_y_input", "value"),
               State("toggle_map", "on")],
              prevent_initial_call=True)
def def_relocate(n_clicks, x_input, y_input, on):
    global vehicle_dict, total_distance, vehicle_startlocations, no_vehicles, order_dict_with_time
    global restaurant_locations, start_of_day_time, total_revenue, date_selected
    if n_clicks is None:
        raise PreventUpdate

    current_vehicle = sim.current_vehicle
    current_time = sim.current_time
    start_location = sim.current_location.copy()
    vehicle_startlocations[current_vehicle] = start_location
    vehicle_startlocations = vehicle_startlocations.copy()

    sim.relocate([x_input, y_input])

    order_velocity, quantile = get_order_velocity_and_quantile()
    vehicle_startlocations[sim.current_vehicle] = sim.current_location
    vehicle_dict = update_vehicle_dict(vehicle_dict, start_location, current_time, current_vehicle)
    start_node, end_node = nav.find_nodes([start_location, [x_input, y_input]])
    total_distance += nav.find_routes(start_node, end_node)[1]

    vehicles_with_orders_and_rewards = algorithm.algorithm_4_dash(sim.vehicle_orders, nav, sim, restaurant_data,
                                                                  heatmap, vehicle_dict, no_vehicles, 8,
                                                                  2, 0, vehicle_distances, "")
    order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, sim.current_time - current_time, False)

    order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time = update_everything(
        vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time)

    income_velocity = total_revenue*60*60/(no_vehicles*(sim.current_time - start_of_day_time))

    return (vehicle_info_pd.to_dict('records'),
            str(dt.timedelta(seconds=sim.current_time)),
            list(sim.orders.keys()),
            f'{total_distance/1000:.2f} km',
            reward_info_pd.to_dict('records'),
            order_info_pd.to_dict('records'),
            map_html,
            sim.current_vehicle,
            f'€{income_velocity:.2f}/hour/vehicle',
            f'{order_velocity:.1f}',
            f'{quantile:.1f}')

'''
callback def_relocate_to_nearest_restaurant activates when the 'relocate to nearest restaurant' button is clicked. 
This function determines the location of the nearest restaurant and calls sim.relocate with that location.
'''
@app.callback([Output("vehicle_table", "data", allow_duplicate=True),
               Output("time", "children", allow_duplicate=True),
               Output("order-dropdown", "options", allow_duplicate=True),
               Output("km_travelled_block", "children", allow_duplicate=True),
               Output('combi_table', 'data', allow_duplicate=True),
               Output('order_table', 'data', allow_duplicate=True),
               Output("map", "src", allow_duplicate=True),
               Output("chosen_vehicle", "children", allow_duplicate=True),
               Output("income_velocity", "children", allow_duplicate=True),
               Output("order_velocity", "children", allow_duplicate=True),
               Output("quantile", "children", allow_duplicate=True)],
              [Input("relocate_to_restaurant", "n_clicks")],
              [State("toggle_map", "on")],
              prevent_initial_call=True)
def def_relocate_to_nearest_restaurant(n_clicks, on):
    global vehicle_dict, total_distance, vehicle_startlocations, no_vehicles, order_dict_with_time
    global restaurant_locations, start_of_day_time, total_revenue, date_selected
    if n_clicks is None:
        raise PreventUpdate

    current_vehicle = sim.current_vehicle
    current_time = sim.current_time
    start_location = sim.current_location.copy()
    vehicle_startlocations[current_vehicle] = start_location
    vehicle_startlocations = vehicle_startlocations.copy()

    loc_nearest_res, distance_to_nearest_res = find_nearest_restaurant(sim.current_location)

    sim.relocate(loc_nearest_res)

    order_velocity, quantile = get_order_velocity_and_quantile()
    vehicle_startlocations[sim.current_vehicle] = sim.current_location
    vehicle_dict = update_vehicle_dict(vehicle_dict, start_location, current_time, current_vehicle)
    total_distance += distance_to_nearest_res

    vehicles_with_orders_and_rewards = algorithm.algorithm_4_dash(sim.vehicle_orders, nav, sim, restaurant_data,
                                                                  heatmap, vehicle_dict, no_vehicles, 8,
                                                                  2, 0, vehicle_distances, "")
    order_dict_with_time = sf.update_order_dict(sim.orders, order_dict_with_time, sim, sim.current_time - current_time, False)

    order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time = update_everything(
        vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time)

    income_velocity = total_revenue*60*60/(no_vehicles*(sim.current_time - start_of_day_time))

    return (vehicle_info_pd.to_dict('records'),
            str(dt.timedelta(seconds=sim.current_time)),
            list(sim.orders.keys()),
            f'{total_distance/1000:.2f} km',
            reward_info_pd.to_dict('records'),
            order_info_pd.to_dict('records'),
            map_html,
            sim.current_vehicle,
            f'€{income_velocity:.2f}/hour/vehicle',
            f'{order_velocity:.1f}',
            f'{quantile:.1f}')

'''
callback def_initialize activates when a date is selected, a number of vehicles for the day is inserted and the 'start' 
button is clicked. This function fills the vehicle_dict dictionary and updates everything accordingly.
'''
@app.callback([Output("time", "children", allow_duplicate=True),
               Output("vehicle_table", "data", allow_duplicate=True),
               Output("revenue_block", "children", allow_duplicate=True),
               Output("order-dropdown", "options", allow_duplicate=True),
               Output("num_orders_block", "children", allow_duplicate=True),
               Output('combi_table', 'data', allow_duplicate=True),
               Output('order_table', 'data', allow_duplicate=True),
               Output("map", "src", allow_duplicate=True),
               Output("chosen_vehicle", "children", allow_duplicate=True),
               Output("order_velocity", "children", allow_duplicate=True),
               Output("income_velocity", "children", allow_duplicate=True),
               Output("km_travelled_block", "children", allow_duplicate=True),
               Output("quantile", "children", allow_duplicate=True)],
              [Input('initialize_input', 'n_clicks')],
              [State("no_vehicles", "value"),
               State("date", "date"),
               State("toggle_map", "on")],
              prevent_initial_call=True)
def def_initialize(n_clicks, value, date, on):
    global vehicle_dict, sim, no_orders, vehicle_startlocations, no_vehicles, restaurant_locations
    global start_of_day_time, order_dict_with_time, date_selected
    no_vehicles = value
    year = int(date.split("-")[0])
    month = int(date.split("-")[1])
    day = int(date.split("-")[2])
    date_selected = dt.datetime(year, month, day)
    sim = Simulator(year, month, day)
    sim.reset(n_vehicles = no_vehicles)

    start_of_day_time = sim.current_time

    order_velocity, quantile = get_order_velocity_and_quantile()

    vehicle_dict = {}
    for id in range(no_vehicles):
        if id == sim.current_vehicle:
            vehicle_dict[id] = ("Available", sim.vehicle_locations[id], dt.timedelta(seconds=sim.current_time), 0)
        else:
            vehicle_dict[id] = ("Unavailable", sim.vehicle_locations[id], dt.timedelta(seconds=sim.current_time), 0)

    for id in range(no_vehicles):
        vehicle_startlocations[id] = sim.vehicle_locations[id]

    vehicle_dict = update_vehicle_dict(vehicle_dict, sim.current_location, sim.current_time, sim.current_vehicle)
    vehicles_with_orders_and_rewards = algorithm.algorithm_4_dash(sim.vehicle_orders, nav, sim, restaurant_data,
                                                                  heatmap, vehicle_dict, no_vehicles, 8,
                                                                  2, 0, vehicle_distances, "")
    order_dict_with_time = sf.update_order_dict(sim.orders, dict(), sim, 0, True)

    order_info_pd, vehicle_info_pd, reward_info_pd, restaurant_locations, map_html, order_dict_with_time = (
        update_everything(vehicles_with_orders_and_rewards, vehicle_dict, restaurant_locations, vehicle_startlocations, on, order_dict_with_time))

    return [str(dt.timedelta(seconds=sim.current_time)),
            vehicle_info_pd.to_dict('records'),
            f'€{total_revenue:.2f}',
            list(sim.orders.keys()),
            no_orders,
            reward_info_pd.to_dict('records'),
            order_info_pd.to_dict('records'),
            map_html,
            sim.current_vehicle,
            f'{order_velocity:.1f}',
            f'€{0:.2f}/hour/vehicle',
            f'{0/1000:.2f} km',
            f'{quantile:.1f}']
'''
callback update_map_callback activates when the 'map' toggle is changed. This callback shows the map with the current 
information.
'''
@callback(Output('map', 'src', allow_duplicate=True),
    Input('toggle_map', 'on'), prevent_initial_call=True)
def update_map_callback(on):
    global vehicle_startlocations, restaurant_locations
    x = "{}".format(on)
    if x == "True":
        map_html = update_map(vehicle_startlocations, restaurant_locations)
        return map_html

'''
These lines import the necessary files. Then the app is run.
'''
if __name__ == '__main__':
    heatmap = sf.import_json_file("Data/heatmap_v1.json")
    quantile_weekday = sf.import_json_file("data/quantile_weekday.json")
    restaurant_data = sf.import_json_file("Data/restaurant_data_v1.json")
    nav = Navigator("data/paris_map.txt")
    vehicle_distances = sf.import_pickle_file('data/area_distances_dict.pkl')
    order_velocity_data = sf.import_json_file('data/order_velocity_means.json')
    print("-" * 125, end="\n\n")
    app.run(debug=True, port=8045)