"""
Generates explanations using decision trees
"""

import collections
from typing import Sequence, List, Tuple, Union
from unittest import skip
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as mplAnimate
from IPython.core.display import HTML
import gym
import numpy as onp
import pydotplus
from pydotplus import Dot
from sklearn import tree
from sklearn.tree import BaseDecisionTree

# Source: https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
env_ram_objects = {
    "Asteroids": dict(enemy_asteroids_y=[3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19],
                      enemy_asteroids_x=[21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37],
                      player_x=73,
                      player_y=74,
                      num_lives_direction=60,
                      player_score_high=61,
                      player_score_low=62,
                      player_missile_x1=83,
                      player_missile_x2=84,
                      player_missile_y1=86,
                      player_missile_y2=87,
                      player_missile1_direction=89,
                      player_missile2_direction=90),

    "BattleZone": dict(  # red_enemy_x=75,
        blue_tank_facing_direction=46,  # 17 left 21 forward 29 right
        blue_tank_size_y=47,  # tank gets larger as it gets closer
        blue_tank_x=48,
        blue_tank2_facing_direction=52,
        blue_tank2_size_y=53,
        blue_tank2_x=54,
        num_lives=58,
        missile_y=105,
        compass_needles_angle=84,
        angle_of_tank=4,  # as shown by what the mountains look like
        left_tread_position=59,  # got to mod this number by 8 to get unique values
        right_tread_position=60,  # got to mod this number by 8 to get unique values
        crosshairs_color=108,  # 0 if black 46 if yellow
        score=29),

    "Berzerk": dict(player_x=19,
                    player_y=11,
                    player_direction=14,
                    player_missile_x=22,
                    player_missile_y=23,
                    player_missile_direction=21,
                    robot_missile_direction=26,
                    robot_missile_x=29,
                    robot_missile_y=30,
                    num_lives=90,
                    robots_killed_count=91,
                    game_level=92,
                    enemy_evilOtto_x=46,
                    enemy_evilOtto_y=89,
                    enemy_robots_x=range(65, 73),
                    enemy_robots_y=range(56, 65),
                    player_score=range(93, 96)),

    "Bowling": dict(ball_x=30,
                    ball_y=41,
                    player_x=29,
                    player_y=40,
                    frame_number_display=36,
                    pin_existence=range(57, 67),
                    score=33),

    "Boxing": dict(player_x=32,
                   player_y=34,
                   enemy_x=33,
                   enemy_y=35,
                   enemy_score=19,
                   clock=17,
                   player_score=18),

    "Breakout": dict(ball_x=99,
                     ball_y=101,
                     player_x=72,
                     blocks_hit_count=77,
                     block_bit_map=range(30),  # see breakout bitmaps tab
                     score=84),  # 5 for each hit

    "DemonAttack": dict(level=62,
                        player_x=22,
                        enemy_x1=17,
                        enemy_x2=18,
                        enemy_x3=19,
                        missile_y=21,
                        enemy_y1=69,
                        enemy_y2=70,
                        enemy_y3=71,
                        num_lives=114),

    "Freeway": dict(player_y=14,
                    score=103,
                    enemy_car_x=range(108, 118)),  # which lane the car collided with player

    "Frostbite": dict(
        top_row_iceflow_x=34,
        second_row_iceflow_x=33,
        third_row_iceflow_x=32,
        fourth_row_iceflow_x=31,
        enemy_bear_x=104,
        num_lives=76,
        igloo_blocks_count=77,  # 255 is none and 15 is all "
        enemy_x=range(84, 88),  # 84  bottom row -   87  top row
        player_x=102,
        player_y=100,
        player_direction=4,
        score=[72, 73, 74]),

    "Hero": dict(player_x=27,
                 player_y=31,
                 power_meter=43,
                 room_number=28,
                 level_number=117,
                 dynamite_count=50,
                 score=[56, 57]),

    "MontezumaRevenge": dict(room_number=3,
                             player_x=42,
                             player_y=43,
                             player_direction=52,
                             # 72:  facing left,
                             # 40:  facing left, climbing down ladder/rope
                             # 24:  facing left, climbing up ladder/rope
                             # 128: facing right
                             # 32:  facing right, climbing down ladder/rope,
                             # 16: facing right climbing up ladder/rope
                             enemy_skull_x=47,
                             enemy_skull_y=46,
                             key_monster_x=44,
                             key_monster_y=45,
                             level=57,
                             num_lives=58,
                             items_in_inventory_count=61,
                             room_state=62,
                             score_0=19,
                             score_1=20,
                             score_2=21),

    "MsPacman": dict(enemy_sue_x=6,
                     enemy_inky_x=7,
                     enemy_pinky_x=8,
                     enemy_blinky_x=9,
                     enemy_sue_y=12,
                     enemy_inky_y=13,
                     enemy_pinky_y=14,
                     enemy_blinky_y=15,
                     player_x=10,
                     player_y=16,
                     fruit_x=11,
                     fruit_y=17,
                     ghosts_count=19,
                     player_direction=56,
                     dots_eaten_count=119,
                     player_score=120,
                     num_lives=123),

    "Pitfall": dict(player_x=97,  # 8-148
                    player_y=105,  # 21-86 except for when respawning then 0-255 with confusing wraparound
                    enemy_logs_x=98,  # 0-160
                    enemy_scorpion_x=99,
                    # player_y_on_ladder= 108, # 0-20
                    # player_collided_with_rope= 5, #yes if bit 6 is 1
                    bottom_of_rope_y=18,  # 0-20 varies even when you can't see rope
                    clock_sec=89,
                    clock_min=88),

    "Pong": dict(player_y=51,
                 player_x=46,
                 enemy_y=50,
                 enemy_x=45,
                 ball_x=49,
                 ball_y=54,
                 enemy_score=13,
                 player_score=14),

    "PrivateEye": dict(player_x=63,
                       player_y=86,
                       room_number=92,
                       clock=[67, 69],
                       player_direction=58,
                       score=[73, 74],
                       dove_x=48,
                       dove_y=39),

    "Qbert": dict(player_x=43,
                  player_y=67,
                  player_column=35,
                  red_enemy_column=69,
                  green_enemy_column=105,
                  score=[89, 90, 91],  # binary coded decimal score
                  tile_color=[21,  # row of 1
                              52, 54,  # row of 2
                              83, 85, 87,  # row of 3
                              98, 100, 102, 104,  # row of 4
                              1, 3, 5, 7, 9,  # row of 5
                              32, 34, 36, 38, 40, 42]),  # row of 6

    "Riverraid": dict(player_x=51,
                      missile_x=117,
                      missile_y=50,
                      fuel_meter_high=55,  # high value displayed
                      fuel_meter_low=56),  # low value

    "Seaquest": dict(enemy_obstacle_x=range(30, 34),
                     player_x=70,
                     player_y=97,
                     diver_or_enemy_missile_x=range(71, 75),
                     player_direction=86,
                     player_missile_direction=87,
                     oxygen_meter_value=102,
                     player_missile_x=103,
                     score=[57, 58],
                     num_lives=59,
                     divers_collected_count=62),

    "Skiing": dict(player_x=25,
                   clock_m=104,
                   clock_s=105,
                   clock_ms=106,
                   score=107,
                   object_y=range(87, 94)),  # object_y_1 is y position of whatever topmost object on the screen is

    "SpaceInvaders": dict(invaders_left_count=17,
                          player_score=104,
                          num_lives=73,
                          player_x=28,
                          enemies_x=26,
                          missiles_y=9,
                          enemies_y=24),

    "Tennis": dict(enemy_x=27,
                   enemy_y=25,
                   enemy_score=70,
                   ball_x=16,
                   ball_y=17,
                   player_x=26,
                   player_y=24,
                   player_score=69),

    "Venture": dict(sprite0_y=20,
                    sprite1_y=21,
                    sprite2_y=22,
                    sprite3_y=23,
                    sprite4_y=24,
                    sprite5_y=25,
                    sprite0_x=79,
                    sprite1_x=80,
                    sprite2_x=81,
                    sprite3_x=82,
                    sprite4_x=83,
                    sprite5_x=84,
                    player_x=85,
                    player_y=26,
                    current_room=90,  # The number of the room the player is currently in 0 to 9_
                    num_lives=70,
                    score_1_2=71,
                    score_3_4=72),

    "VideoPinball": dict(ball_x=67,
                         ball_y=68,
                         player_left_paddle_y=98,
                         player_right_paddle_y=102,
                         score_1=48,
                         score_2=50),

    "YarsRevenge": dict(player_x=32,
                        player_y=31,
                        player_missile_x=38,
                        player_missile_y=37,
                        enemy_x=43,
                        enemy_y=42,
                        enemy_missile_x=47,
                        enemy_missile_y=46)
}

WhyNodeExplanation = collections.namedtuple('WhyNodeExplanation',
                                            ['node_id', 'feature_id', 'feature_name', 'feature_value',
                                             'threshold_sign', 'threshold_value'])

WhyNotNodeExplanation = collections.namedtuple('WhyNotNodeExplanation',
                                               ['node_id', 'feature_id', 'feature_name', 'feature_value',
                                                'counterfactual', 'threshold_sign', 'threshold_value'])


def get_env_feature_labels(env_name: str) -> Tuple[onp.ndarray, onp.ndarray]:
    """
    Returns an array of environment feature names

    :param env_name: The environment name
    :return: Array of environment feature names
    """
    feature_names = [f'feature {pos}' for pos in range(128)]
    has_feature = onp.zeros(128, dtype=onp.bool)
    for feature_name, indices in env_ram_objects[env_name].items():
        if isinstance(indices, Sequence):
            for index in indices:
                feature_names[index] = feature_name.replace('_', ' ').title()
                has_feature[index] = True
        elif isinstance(indices, int):
            feature_names[indices] = feature_name.replace('_', ' ').title()
            has_feature[indices] = True
        else:
            raise Exception((feature_name, type(indices), indices))

    return onp.array(feature_names), onp.where(has_feature)[0]


def is_necessary_decision(raw_tree, node_id: int, obs: onp.ndarray, action: int) -> bool:
    """
    Check if the decision node is necessary for the action output

    :param raw_tree: The raw tree of a decision tree
    :param node_id: The current node id
    :param obs: The environment observation
    :param action: The expected action
    :return: If the decision node is necessary for the output
    """
    while True:
        if raw_tree.children_left[node_id] == -1 and raw_tree.children_right[node_id] == -1:
            # print(f'\tLeft node {node_id}')
            return onp.argmax(raw_tree.value[node_id]) != action

        if obs[raw_tree.feature[node_id]] <= raw_tree.threshold[node_id]:
            node_id = raw_tree.children_left[node_id]
            # print(f'\tLeft node {node_id}')
        else:
            node_id = raw_tree.children_right[node_id]
            # print(f'\tRight node {node_id}')

def animate_observations(obs:onp.ndarray , savePath:str, frame_interval: int = 200, figsize: Tuple[int, int] = (8, 8),
                         fig_title: str = None, return_html_animation: bool = False):
    fig, ax = plt.subplots(figsize=figsize)
    if fig_title:
        fig.suptitle(fig_title)

    ax.axis('off')
    obs_plot = ax.imshow(obs[0, :, :, -1])
    ax.set_xlabel('0')
    onp.asarray(obs)
    def _animate_time_step(time_step: int):
        obs_plot.set_data(obs[time_step, :, :, -1])
        ax.set_xlabel(time_step)
        return [obs_plot]

    gameplay = mplAnimate.FuncAnimation(fig, _animate_time_step, frames=len(obs), interval=frame_interval)
    print(matplotlib.__version__)
    videoWriter=mplAnimate.FFMpegWriter(fps=1)
    gameplay.save(savePath,writer=videoWriter)
    if return_html_animation:
        plt.close()
        return HTML(gameplay.to_html5_video())
    else:
        return fig, ax, gameplay
def dt_why_explanation(obs: onp.ndarray, decision_tree: BaseDecisionTree, env_name: str,
                       remove_unnecessary_nodes: bool = True) -> str:
    """
    Generates a list of explanations for a policy tree (can either be the action dt or q-values dt)

    :param obs: environment observations
    :param decision_tree: The decision path to explain
    :param env_name: the environment name
    :param remove_unnecessary_nodes: If to remove unnecessary nodes from the explanation
    :return: Decision tree explanation using explanation template
    """
    explanations=[]
    for  o in obs:
        print(o.shape)
        assert o.shape == (128,)
        decision_path = decision_tree.decision_path(o.reshape(1, -1))
        print("reshaped (1,-1)",o.reshape(1, -1).shape)
        action = int(decision_tree.predict(o.reshape(1, -1))[0]) #actions chosen is always the first element in RAM obs
        print("action",action)
        raw_tree = decision_tree.tree_
        tree_features, tree_thresholds = raw_tree.feature, raw_tree.threshold

        node_path = []
        
        feature_names, _ = get_env_feature_labels(env_name)
        index=0
        for node_id in decision_path.indices[:-1]:
            # print('Node:', node_id)
            threshold_sign = '<=' if o[tree_features[node_id]] <= tree_thresholds[node_id] else '>'
            feature_name = feature_names[tree_features[node_id]]

            if remove_unnecessary_nodes:
                if o[tree_features[node_id]] <= tree_thresholds[node_id]:
                        counterfactual_node_id = raw_tree.children_right[node_id]
                else:
                        counterfactual_node_id = raw_tree.children_left[node_id]

                    # print(counterfactual_node_id)
                if counterfactual_node_id == -1 or is_necessary_decision(raw_tree, counterfactual_node_id, o, action):
                        node_path.append(WhyNodeExplanation(node_id, tree_features[node_id], feature_name,
                                                            o[tree_features[node_id]], threshold_sign,
                                                            tree_thresholds[node_id]))

            if len(node_path) == 1:
                explanation = f'As the {node_path[0].feature_name} {node_path[0].threshold_sign} {node_path[0].threshold_value}'
                explanation += f' then {gym.make(env_name + "-v0").get_action_meanings()[action].lower().title()} action is taken'
                explanations.append(explanation)
            elif len(node_path) > 1:
                print("node path", node_path)
                print("node path shape", onp.shape(node_path))
                print("lenght nodepath ",len(node_path))
                # TODO: Post-processing on the node path to group similar parts together
                explanation = 'As the ' + ', '.join([
                    f'{node.feature_name} {node.threshold_sign} {node.threshold_value}' for node in node_path[:-1]
                ]) + f' and {node_path[-1].feature_name} {node_path[-1].threshold_sign} {node_path[-1].threshold_value}'
                explanation += f' then {gym.make(env_name + "-v0").get_action_meanings()[action].lower().title()} action is taken'
                explanations.append(explanation)
            else:
                explanations.append("Empty nodepath")
            print("explantion of action ",index,": ",explanation)
            print("explanations list shape", onp.shape(explanations))
            index+=1
    return explanations


def dt_why_not_explanation(obs: onp.ndarray, decision_tree: BaseDecisionTree, env_name: str,
                           counterfactual_action: int, node_depth_weight: float = 0.8,
                           return_route: bool = False) -> Union[str, Tuple[str, List[Tuple[int, bool]]]]:
    """
    Explanation for why not the decision tree takes an action

    :param obs: Current environment observations
    :param decision_tree: The decision tree
    :param env_name: The environment name
    :param counterfactual_action: The counterfactual action
    :param node_depth_weight:
    :param return_route:
    :return:
    """
    assert obs.shape == (128,)
    assert 0 <= counterfactual_action < gym.make(f'{env_name}-v0').action_space.n, \
        f'0 <= {counterfactual_action} <= {gym.make(f"{env_name}-v0").action_space.n}'
    assert 0 < node_depth_weight <= 1

    raw_tree = decision_tree.tree_
    tree_features, tree_thresholds = raw_tree.feature, raw_tree.threshold

    def why_not_route(_node_id: int, cost: float, route: List[int], depth: int) -> Tuple[float, List[int]]:
        """
        Calculates the minimum cost of counterfactual decisions to reach an action

        :param _node_id: The current node id
        :param cost: The current code
        :param route: The current route of node ids
        :param depth: The current depth
        :return: The future minimum cost and its route
        """
        if raw_tree.children_left[_node_id] == -1 and raw_tree.children_right[_node_id] == -1:
            if onp.argmax(raw_tree.value[_node_id]) == counterfactual_action:
                return cost, route[:] + [_node_id]
            else:
                return 1000, route[:] + [_node_id]
        else:
            node_decision = obs[tree_features[_node_id]] <= tree_thresholds[_node_id]
            left_cost, left_route = why_not_route(raw_tree.children_left[_node_id],
                                                  cost + pow(node_depth_weight, depth) * (not node_decision),
                                                  route[:] + [_node_id], depth + 1)
            right_cost, right_route = why_not_route(raw_tree.children_right[_node_id],
                                                    cost + pow(node_depth_weight, depth) * node_decision,
                                                    route[:] + [_node_id], depth + 1)

            if left_cost < right_cost:
                return left_cost, left_route
            elif left_cost == right_cost:
                if len(left_route) < len(right_route):
                    return left_cost, left_route
                else:
                    return right_cost, right_route
            else:
                assert left_cost > right_cost
                return right_cost, right_route

    minimum_cost, minimum_route = why_not_route(0, 0, [], 0)
    node_path = []
    env_feature_names, _ = get_env_feature_labels(env_name)
    for pos, node_id in enumerate(minimum_route[:-1]):
        if obs[tree_features[node_id]] <= tree_thresholds[node_id]:
            counterfactual, threshold_sign = raw_tree.children_left[node_id] == minimum_route[pos + 1], '<='
        else:
            counterfactual, threshold_sign = raw_tree.children_right[node_id] == minimum_route[pos + 1], '>'

        feature_name = env_feature_names[tree_features[node_id]]
        node_path.append(WhyNotNodeExplanation(node_id, tree_features[node_id], feature_name,
                                               obs[tree_features[node_id]], counterfactual, threshold_sign,
                                               tree_thresholds[node_id]))

    # TODO: Current just returning the counterfactual nodes not both counterfactual and true values
    cf_node_path = [node for node in node_path if node.counterfactual]
    if len(cf_node_path) == 1:
        explanation = f'As the {cf_node_path[0].feature_name} ' \
                      f'{cf_node_path[0].threshold_sign} {cf_node_path[0].threshold_value}'
    else:
        # TODO: Post-processing on the node path to group similar parts together
        explanation = f'As the ' + ', '.join([
            f'{node.feature_name} {node.threshold_sign} {node.threshold_value}' for node in cf_node_path[:-1]
        ]) + f' and {cf_node_path[-1].feature_name} {cf_node_path[-1].threshold_sign} ' \
             f'{cf_node_path[-1].threshold_value}'

    env = gym.make(env_name + "-v0")
    explanation += f' then {env.get_action_meanings()[counterfactual_action].lower().title()} action is not taken'

    if return_route:
        return explanation, [(node.node_id, node.counterfactual) for node in node_path] + [(minimum_route[-1], False)]
    else:
        return explanation


def visualise_decision_tree(decision_tree: BaseDecisionTree, env_name: str, filename: str = None):
    """
    Visualise a decision tree

    :param decision_tree: The decision to visualise
    :param env_name: The environment name
    :param filename: The image filename
    :return: The resulting graph
    """
    dot_data = tree.export_graphviz(decision_tree, class_names=gym.make(f'{env_name}-v0').get_action_meanings(),
                                    filled=True, rounded=True, special_characters=True,
                                    feature_names=get_env_feature_labels(env_name)[0])
    graph = pydotplus.graph_from_dot_data(dot_data)

    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:  # Arrows
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            if len(labels) == 4:
                node.set('label', f'<{labels[3]}')
                node.set_fillcolor('lightblue')
            elif len(labels) == 5:
                node.set('label', f'{labels[0]}>')
                node.set_fillcolor('white')
            else:
                print(labels)

    if filename:
        graph.write_png(filename)
    return graph


def visualise_why_explanation(obs: onp.ndarray, decision_tree: BaseDecisionTree, env_name: str,
                              filename: str = None) -> Dot:
    """
    Visualise the why decision path by the decision tree

    :param obs: The observation to find the decision path
    :param decision_tree: The foundational decision path
    :param env_name: Environment name
    :param filename: The image filename
    :return: The resulting graph
    """
    graph = visualise_decision_tree(decision_tree, env_name)

    decision_path = decision_tree.decision_path(obs.reshape(1, -1))[0]
    for node_id in decision_path.indices:
        graph.get_node(str(node_id))[0].set_fillcolor('green')

    if filename:
        graph.write_png(filename)
    return graph


def visualise_why_not_explanation(decision_path: List[Tuple[int, bool]], decision_tree: BaseDecisionTree, env_name: str,
                                  filename: str = None) -> Dot:
    """
    Visualise the why not decision path by the decision tree

    :param decision_path: The why not decision path with each node id and if it is a counterfactual
    :param decision_tree: The decision tree
    :param env_name: The environment for the feature labels
    :param filename: Save filename
    :return: The resulting graph
    """
    graph = visualise_decision_tree(decision_tree, env_name)

    for node_id, counterfactual in decision_path:
        graph.get_node(str(node_id))[0].set_fillcolor('red' if counterfactual else 'green')

    if filename:
        graph.write_png(filename)
    return graph
