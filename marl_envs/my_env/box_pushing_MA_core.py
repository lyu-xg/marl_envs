#!/usr/bin/python

import numpy as np
import gym
import IPython

from gym.utils import seeding

#DIRECTION = [(0.0,1.0), (1.0,0.0), (0.0,-1.0), (-1.0,0.0)]
DIRECTION = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.0, -1.0],
                      [-1.0, 0.0]])

COST = np.array([[0.0, 0.1, 0.2, 0.1],
                 [0.1, 0.0, 0.1, 0.2],
                 [0.2, 0.1, 0.0, 0.1],
                 [0.1, 0.2, 0.1, 0.0]])

# COST = np.array([[0.0, 1.0, 2.0, 1.0],
#                  [1.0, 0.0, 1.0, 2.0],
#                  [2.0, 1.0, 0.0, 1.0],
#                  [1.0, 2.0, 1.0, 0.0]])



class Agent(object):

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 init_ori,
                 beliefwaypoints,
                 MAs,
                 speed=1.0):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0,0.0])

        self.BWPs = beliefwaypoints
        self.cur_BWP = None

        self.macro_actions = MAs
        self.cur_action = None
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        self.speed = speed

    def step(self, action, boxes):

        assert action < len(self.macro_actions), "The action received is out of range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        if action < 2:
            bwpterm_idx = self.cur_action.ma_bwpterm
            dist = round(self._get_dist(self.BWPs[bwpterm_idx].xcoord, self.BWPs[bwpterm_idx].ycoord), 2)
            if dist <= self.speed:
                self.xcoord = self.BWPs[bwpterm_idx].xcoord
                self.ycoord = self.BWPs[bwpterm_idx].ycoord
                self.cur_BWP = self.BWPs[bwpterm_idx]   # need ?
                self.cur_action_time_left = 0.0
                self.cur_action_done = True
                self.ori = 0
            else:
                delta_x = self.speed / dist * (self.BWPs[bwpterm_idx].xcoord - self.xcoord)
                delta_y = self.speed / dist * (self.BWPs[bwpterm_idx].ycoord - self.ycoord)
                self.xcoord += delta_x
                self.ycoord += delta_y
                self.cur_action_time_left = dist - self.speed
                self.direct = self._get_dir(self.BWPs[bwpterm_idx])

        elif action == 2:
            if (self.ori == 0 and self.ycoord == 7.5) or \
                    (self.ori == 1 and self.xcoord == 7.5) or \
                    (self.ori == 2 and self.ycoord == 0.5) or \
                    (self.ori == 3 and self.xcoord == 0.5):
                        reward += -5.0
                        self.cur_action_time_left = 0.0
                        self.cur_action_done = True
            else:
                pushing_box = False
                direction = DIRECTION[self.ori]
                self.xcoord += direction[0]
                self.ycoord += direction[1]

                # check if box is pushed
                for box in boxes:
                    if (box.xcoord == self.xcoord and box.ycoord == self.ycoord) or \
                            ((box.xcoord == self.xcoord-0.5 or box.xcoord == self.xcoord+0.5) and box.ycoord == self.ycoord):
                        if self.ori == 0 and box.idx != 2:
                            pushing_box = True
                            box.xcoord += direction[0]
                            box.ycoord += direction[1]
                            if box.ycoord == 7.5:
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                        else:
                            self.xcoord -= direction[0]
                            self.ycoord -= direction[1]
                            self.cur_action_time_left = 0.0
                            self.cur_action_done = True
                            reward += -5.0

                # check if push action is done
                if not pushing_box:
                    if (self.ori == 0 and self.ycoord == 7.5) or \
                            (self.ori == 1 and self.xcoord == 7.5) or \
                            (self.ori == 2 and self.ycoord == 0.5) or \
                            (self.ori == 3 and self.xcoord == 0.5):
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                    else:
                        self.cur_action_time_left = -1.0
        elif action == 3:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1
            self.cur_action_done = True
        
        elif action == 4:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1
            self.cur_action_done = True
        elif action == 5:
            self.cur_action_done = True
 
        return reward
    
    def _get_dist(self, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - self.xcoord)**2 + (g_ycoord - self.ycoord)**2)

    def _get_dir(self, BWP):
        v = np.array([BWP.xcoord-self.xcoord, BWP.ycoord-self.ycoord])
        return v / np.linalg.norm(v)

class Box(object):
    
    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 size_h,
                 size_w):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.h = size_h
        self.w = size_w

class BeliefWayPoint(object):

    """Properties for a waypoint in the 2D sapce"""

    def __init__(self,
                 name,
                 idx,
                 xcoord,
                 ycoord):
        
        self.name = name
        self.idx = idx
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.occupied = False

class MacroAction(object):

    """Properties for a macro_action"""

    def __init__(self, 
                 name,
                 idx,
                 expected_t_cost=None,
                 std=None,
                 ma_bwpterm=None):

        self.name = name  # the name of this macro-action
        self.idx = idx    # the index of this macro-action
        self.expected_t_cost = expected_t_cost   # None is for moving action. When it is done depends on the specify speed.
        self.std = std
        if std is None:
            self.real_t_cost = expected_t_cost   # the time cost of finishing this macro-action
        else:
            self.real_t_cost = np.random.normal(expected_t_cost, std)
        self.ma_bwpterm = ma_bwpterm  # used for moving action to indicate at which belief waypoint this macro-action will be terminated,
                                      # None means the terminate belief waypoint is same as where the action is initialized.

    @property
    def t_cost(self):
        if self.std is None:
            return self.expected_t_cost   # the time cost of finishing this macro-action
        else:
            return round(np.random.normal(self.expected_t_cost, self.std),1)   # resample a time cost for the macro-action
 
class astar_Agent(object):

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 init_ori,
                 beliefwaypoints,
                 MAs,
                 speed=1.0):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0,0.0])

        self.BWPs = beliefwaypoints
        self.cur_BWP = None

        self.macro_actions = MAs
        self.cur_action = None
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        self.speed = speed

    def step(self, action, boxes):

        assert action < len(self.macro_actions), "The action received is out of range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        if action < 2:
            bwpterm_idx = self.cur_action.ma_bwpterm
            self.move(self.BWPs[bwpterm_idx], boxes)

        elif action == 2:
            if (self.ori == 0 and self.ycoord == 7.5) or \
                    (self.ori == 1 and self.xcoord == 7.5) or \
                    (self.ori == 2 and self.ycoord == 0.5) or \
                    (self.ori == 3 and self.xcoord == 0.5):
                        reward += -5.0
                        self.cur_action_time_left = 0.0
                        self.cur_action_done = True
            else:
                pushing_box = False
                direction = DIRECTION[self.ori]
                self.xcoord += direction[0]
                self.ycoord += direction[1]

                # check if box is pushed
                for box in boxes:
                    if (box.xcoord == self.xcoord and box.ycoord == self.ycoord) or \
                            ((box.xcoord == self.xcoord-0.5 or box.xcoord == self.xcoord+0.5) and box.ycoord == self.ycoord):
                        if self.ori == 0 and box.idx != 2:
                            pushing_box = True
                            box.xcoord += direction[0]
                            box.ycoord += direction[1]
                            if box.ycoord == 7.5:
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                        else:
                            self.xcoord -= direction[0]
                            self.ycoord -= direction[1]
                            self.cur_action_time_left = 0.0
                            self.cur_action_done = True
                            reward += -5.0

                # check if push action is done
                if not pushing_box:
                    if (self.ori == 0 and self.ycoord == 7.5) or \
                            (self.ori == 1 and self.xcoord == 7.5) or \
                            (self.ori == 2 and self.ycoord == 0.5) or \
                            (self.ori == 3 and self.xcoord == 0.5):
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                    else:
                        self.cur_action_time_left = -1.0
        elif action == 3:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1
            self.cur_action_done = True
        
        elif action == 4:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1
            self.cur_action_done = True
        elif action == 5:
            self.cur_action_done = True
 
        return reward

    def move(self, bwp, boxes):

        agent_p = np.array([self.xcoord, self.ycoord])
        g_p = np.array([bwp.xcoord, bwp.ycoord])
        dist = np.linalg.norm(g_p - agent_p)
        
        if dist < 0.1:
            self.ori = 0
            self.cur_BWP = self.BWPs[bwp.idx]
            self.cur_action_time_left = 0.0
            self.cur_action_done = True
        else:
            moves = DIRECTION + agent_p
            obstacles = [boxes[0].xcoord, boxes[1].xcoord, boxes[2].xcoord+0.5, boxes[2].xcoord-0.5]
            h = np.linalg.norm(g_p-moves, axis=1)
            for idx, move in enumerate(moves):
                if move[1] == boxes[0].ycoord:
                    if move[0] in obstacles:
                        h[idx] = float('inf')
            f = h + COST[self.ori]
            dest_idx = f.argmin()
            if COST[self.ori][dest_idx] == 0.0:
                self.xcoord = moves[dest_idx][0]
                self.ycoord = moves[dest_idx][1]
            elif COST[self.ori][dest_idx] == 0.2:
                self.ori += 1
                if self.ori > 3:
                    self.ori = 0
            else:
                self.ori = dest_idx
            
            if self.xcoord == bwp.xcoord and self.ycoord == bwp.ycoord and self.ori==0:
                self.cur_BWP = self.BWPs[bwp.idx]
                self.cur_action_time_left = 0.0
                self.cur_action_done = True
        
    def _get_dist(self, x, y, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - x)**2 + (g_ycoord - y)**2)

    def _get_dir(self, BWP):
        v = np.array([BWP.xcoord-self.xcoord, BWP.ycoord-self.ycoord])
        return v / np.linalg.norm(v)

class astar_Agent_(object):

    """for box_pushing_harder"""

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 init_ori,
                 beliefwaypoints,
                 MAs,
                 grid_dim,
                 speed=1.0):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0,0.0])

        self.BWPs = beliefwaypoints
        self.cur_BWP = None

        self.macro_actions = MAs
        self.cur_action = None
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        self.speed = speed

        self.xlen, self.ylen = grid_dim

    def step(self, action, boxes):

        assert action < len(self.macro_actions), "The action received is out of range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        if action < 4:
            bwpterm_idx = self.cur_action.ma_bwpterm
            self.move(self.BWPs[bwpterm_idx], boxes)

        elif action == 4:
            if (self.ori == 0 and self.ycoord == self.ylen-0.5) or \
                    (self.ori == 1 and self.xcoord == self.xlen-0.5) or \
                    (self.ori == 2 and self.ycoord == 0.5) or \
                    (self.ori == 3 and self.xcoord == 0.5):
                        reward += -5.0
                        self.cur_action_time_left = 0.0
                        self.cur_action_done = True
            else:
                pushing_box = False
                direction = DIRECTION[self.ori]
                self.xcoord += direction[0]
                self.ycoord += direction[1]

                # check if box is pushed
                for box in boxes:
                    if (box.xcoord == self.xcoord and box.ycoord == self.ycoord) or \
                            ((box.xcoord == self.xcoord-0.5 or box.xcoord == self.xcoord+0.5) and box.ycoord == self.ycoord):
                        if self.ori == 0 and box.idx != 2:
                            pushing_box = True
                            box.xcoord += direction[0]
                            box.ycoord += direction[1]
                            if box.ycoord == self.ylen-0.5:
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                        else:
                            self.xcoord -= direction[0]
                            self.ycoord -= direction[1]
                            self.cur_action_time_left = 0.0
                            self.cur_action_done = True
                            reward += -5.0

                # check if push action is done
                if not pushing_box:
                    if (self.ori == 0 and self.ycoord == self.ylen-0.5) or \
                            (self.ori == 1 and self.xcoord == self.ylen-0.5) or \
                            (self.ori == 2 and self.ycoord == 0.5) or \
                            (self.ori == 3 and self.xcoord == 0.5):
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                    else:
                        self.cur_action_time_left = -1.0
        elif action == 5:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1
            self.cur_action_done = True
        
        elif action == 6:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1
            self.cur_action_done = True
        elif action == 7:
            self.cur_action_done = True
 
        return reward

    def move(self, bwp, boxes):

        agent_p = np.array([self.xcoord, self.ycoord])
        g_p = np.array([bwp.xcoord, bwp.ycoord])
        dist = np.linalg.norm(g_p - agent_p)
        
        if dist < 0.1:
            self.ori = 0
            self.cur_BWP = self.BWPs[bwp.idx]
            self.cur_action_time_left = 0.0
            self.cur_action_done = True
        else:
            moves = DIRECTION + agent_p
            obstacles = [boxes[0].xcoord, boxes[1].xcoord, boxes[2].xcoord+0.5, boxes[2].xcoord-0.5]
            #h = np.linalg.norm(g_p-moves, axis=1)
            h = np.sum(np.abs(g_p-moves), axis=1)
            for idx, move in enumerate(moves):
                if move[1] == boxes[0].ycoord:
                    if move[0] in obstacles:
                        h[idx] = float('inf')
            f = h + COST[self.ori]
            dest_idx = f.argmin()
            if COST[self.ori][dest_idx] == 0.0:
                self.xcoord = moves[dest_idx][0]
                self.ycoord = moves[dest_idx][1]
            elif COST[self.ori][dest_idx] == 0.2:
                self.ori += 1
                if self.ori > 3:
                    self.ori = 0
            else:
                self.ori = dest_idx
            
            if self.xcoord == bwp.xcoord and self.ycoord == bwp.ycoord and self.ori==0:
                self.cur_BWP = self.BWPs[bwp.idx]
                self.cur_action_time_left = 0.0
                self.cur_action_done = True
        
    def _get_dist(self, x, y, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - x)**2 + (g_ycoord - y)**2)

    def _get_dir(self, BWP):
        v = np.array([BWP.xcoord-self.xcoord, BWP.ycoord-self.ycoord])
        return v / np.linalg.norm(v)

class astar_Agent_small(object):

    """for box_pushing_harder"""

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 init_ori,
                 beliefwaypoints,
                 MAs,
                 grid_dim,
                 speed=1.0):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0,0.0])

        self.BWPs = beliefwaypoints
        self.cur_BWP = None

        self.macro_actions = MAs
        self.cur_action = None
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        self.speed = speed

        self.xlen, self.ylen = grid_dim

    def step(self, action, boxes):

        assert action < len(self.macro_actions), "The action received is out of range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        if action < 1:
            bwpterm_idx = self.cur_action.ma_bwpterm
            self.move(self.BWPs[bwpterm_idx], boxes)

        elif action == 1:
            if (self.ori == 0 and self.ycoord == self.ylen-0.5) or \
                    (self.ori == 1 and self.xcoord == self.xlen-0.5) or \
                    (self.ori == 2 and self.ycoord == 0.5) or \
                    (self.ori == 3 and self.xcoord == 0.5):
                        reward += -5.0
                        self.cur_action_time_left = 0.0
                        self.cur_action_done = True
            else:
                pushing_box = False
                direction = DIRECTION[self.ori]
                self.xcoord += direction[0]
                self.ycoord += direction[1]

                # check if box is pushed
                for box in boxes:
                    if (box.xcoord == self.xcoord and box.ycoord == self.ycoord):
                            #((box.xcoord == self.xcoord-0.5 or box.xcoord == self.xcoord+0.5) and box.ycoord == self.ycoord):
                        if self.ori == 0:
                            pushing_box = True
                            box.xcoord += direction[0]
                            box.ycoord += direction[1]
                            if box.ycoord == self.ylen-0.5:
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                        else:
                            self.xcoord -= direction[0]
                            self.ycoord -= direction[1]
                            self.cur_action_time_left = 0.0
                            self.cur_action_done = True
                            #reward += -5.0

                # check if push action is done
                if not pushing_box:
                    if (self.ori == 0 and self.ycoord == self.ylen-0.5) or \
                            (self.ori == 1 and self.xcoord == self.ylen-0.5) or \
                            (self.ori == 2 and self.ycoord == 0.5) or \
                            (self.ori == 3 and self.xcoord == 0.5):
                                self.cur_action_time_left = 0.0
                                self.cur_action_done = True
                    else:
                        self.cur_action_time_left = -1.0
        elif action == 2:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1
            self.cur_action_done = True
        
        elif action == 3:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1
            self.cur_action_done = True
        elif action == 4:
            self.cur_action_done = True
 
        return reward

    def move(self, bwp, boxes):

        agent_p = np.array([self.xcoord, self.ycoord])
        g_p = np.array([bwp.xcoord, bwp.ycoord])
        dist = np.linalg.norm(g_p - agent_p)
        
        if dist < 0.1:
            self.ori = 0
            self.cur_BWP = self.BWPs[bwp.idx]
            self.cur_action_time_left = 0.0
            self.cur_action_done = True
        else:
            moves = DIRECTION + agent_p
            obstacles = [boxes[0].xcoord, boxes[1].xcoord]
            h = np.linalg.norm(g_p-moves, axis=1)
            for idx, move in enumerate(moves):
                if move[1] == boxes[0].ycoord:
                    if move[0] in obstacles:
                        h[idx] = float('inf')
            f = h + COST[self.ori]
            dest_idx = f.argmin()
            if COST[self.ori][dest_idx] == 0.0:
                self.xcoord = moves[dest_idx][0]
                self.ycoord = moves[dest_idx][1]
            elif COST[self.ori][dest_idx] == 0.2:
                self.ori += 1
                if self.ori > 3:
                    self.ori = 0
            else:
                self.ori = dest_idx
            
            if self.xcoord == bwp.xcoord and self.ycoord == bwp.ycoord and self.ori==0:
                self.cur_BWP = self.BWPs[bwp.idx]
                self.cur_action_time_left = 0.0
                self.cur_action_done = True
        
    def _get_dist(self, x, y, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - x)**2 + (g_ycoord - y)**2)

    def _get_dir(self, BWP):
        v = np.array([BWP.xcoord-self.xcoord, BWP.ycoord-self.ycoord])
        return v / np.linalg.norm(v)
