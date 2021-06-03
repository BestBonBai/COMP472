# -------------------------------------------------------
# Assignment (1)
# Written by (Hualin Bai : 40053833)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------

import numpy as np
import math
from heapq import heappush, heappop, heapify
from create_map import MyMap, Grid, Point

class A_Star_Search:
    '''
    class A_Star_Search : implement A* algorithm
    For (Role C)
    :param: start_point : start point class
    :param: end_point : end point
    :param: myMap : a valid map 
    '''

    class PathNode:
        '''
        The class is to store the optimal path for the goal point.
        :param: found path
        :param: total cost : f_cost of the goal point
        '''
        def __init__(self, found_path, total_cost):
            self.found_path = found_path # store the path
            self.total_cost = total_cost # total cost from start to goal point

        def __lt__(self,other):
            '''
            @override compare < method
            compare the total_cost of 2 PathNodes
            '''
            if self.total_cost < other.total_cost:
                return True
            return False

            

    def __init__(self, start_point, end_point, current_map):
        self.start_point = start_point
        self.end_point = end_point
        self.current_map = current_map
        self.open_list = []
        self.close_list = []
        self._INFINUM = 999 # for accessible
        self.path_list = [] # for path
        self.path_nodes_list = [] # for PathNode
        self.goal_points_list = [] # for goal points
    
    def is_valid_point(self, aPoint):
        ''' 
        The method is to check if the point is valid or not 
        :param: aPoint
        :return: True or False
        '''
        if isinstance(aPoint, Point):
            # check if the point is in the map
            if aPoint.x >= 0 and aPoint.x <= self.current_map.xlim \
                and aPoint.y >=0 and aPoint.y <= self.current_map.ylim:
                return True
        print(f'[Error] Point {aPoint.x, aPoint.y} is invalid!')
        return False

    def __is_valid_transform(self, aPoint):
        '''
        The method is to check if the point is needed to transform into other point.
        :param: aPoint
        :return: True or False
        '''
        # check if the point is in the list_points of the map, if so, no need to transform
        for p_i in self.current_map.list_points:
            if p_i == aPoint:
                print(f'[Warning] Point {aPoint.x, aPoint.y} does not need to transform!')
                return False
        return True

    def transform_point(self, aPoint):
        '''
        The method is to transform the point into the required point in the map.
        :param: aPoint
        '''
        if self.is_valid_point(aPoint) and self.__is_valid_transform(aPoint):
            # if the point is inside the cell, 
            # then round into the point to the nearest top-right point of the cell
            for grid_i in self.current_map.list_grids:
                # print(f'[Debug] {grid_i.x_min, grid_i.x_max, grid_i.y_min, grid_i.y_max}')
                if grid_i.x_min < aPoint.x and aPoint.x < grid_i.x_max \
                    and grid_i.y_min < aPoint.y and aPoint.y < grid_i.y_max:
                    # transform the top-right point of the grid
                    __x, __y = grid_i.get_top_right_point()
                    # find it in the list_points 
                    for p_i in self.current_map.list_points:
                        if __x == p_i.x and __y == p_i.y:
                            # assign the aPoint 
                            print(f'[Transform] Point {aPoint.x, aPoint.y} to Point {p_i.x, p_i.y}')
                            aPoint.x = p_i.x
                            aPoint.y = p_i.y
                            aPoint.belong_num_grids = p_i.belong_num_grids
                            return 
            # if not inside the cell, then check the boundary of the grids
            # Case: in the boundary of the column line
            print(f'[Transform] Point is : {aPoint.x, aPoint.y}')
            # print(f'[Debug Float Mod] {np.round(math.fmod(aPoint.x*10, self.current_map.offsetX*10), 1), np.round(math.fmod(aPoint.y*10, self.current_map.offsetY*10), 1)} {np.round((aPoint.y*10 + (self.current_map.offsetY*10 - np.round(aPoint.y*10 % self.current_map.offsetY*10, 1)))/10,1)}')
            # print(f'{np.round(aPoint.x % self.current_map.offsetX, 2), np.round(aPoint.y % self.current_map.offsetY, 2)}')
            # multiply 10 to handle overfloat issue
            if np.round(math.fmod(aPoint.x*10, self.current_map.offsetX*10), 1) == 0.0 and \
                np.round(math.fmod(aPoint.y*10, self.current_map.offsetY*10), 1) != 0.0:
                # round the point's y value
                aPoint.y = np.round((aPoint.y*10 + (self.current_map.offsetY*10 - np.round(math.fmod(aPoint.y*10, self.current_map.offsetY*10), 1)))/10,1)
            # case: in the boundary of the row line
            if np.round(math.fmod(aPoint.x*10, self.current_map.offsetX*10), 1) != 0.0 and \
                np.round(math.fmod(aPoint.y*10, self.current_map.offsetY*10), 1) == 0.0:
                # round the point's y value
                aPoint.x = np.round((aPoint.x*10 + (self.current_map.offsetX*10 - np.round(math.fmod(aPoint.x*10, self.current_map.offsetX*10), 1)))/10,1)
            # find it in the list_points
            for p_i in self.current_map.list_points:
                if aPoint.x == p_i.x and aPoint.y == p_i.y:
                    # assign the aPoint 
                    print(f'[Transform] new Point is {p_i.x, p_i.y}')
                    aPoint.x = p_i.x
                    aPoint.y = p_i.y
                    aPoint.belong_num_grids = p_i.belong_num_grids
                    return
        else:
            print(f'[Warning] Failed Transform point {aPoint.x, aPoint.y}!')
    
    def calc_cost_with_2_points(self, current_point, other_point):
        '''
        The method is to calculate the cost of the 2 points
        :param: current_point
        :param: other point
        :return: cost : between 2 points
        '''
        # get the intersection set of the 2 points' belong_num_grids
        __current_num = self.__get_num_point_in_list(current_point.x, current_point.y)
        __inter_set = self.current_map.list_points[__current_num].belong_num_grids.intersection(other_point.belong_num_grids)
        # print(f'[Debug] belong_num_grids is : {self.current_map.list_points[__current_num].belong_num_grids, other_point.belong_num_grids}')
        # print(f'[News] intersection grid of {current_point.x, current_point.y} and {other_point.x, other_point.y} is : {__inter_set}, length is : {len(__inter_set)}')
        # transfer to list
        __list_inter = list(__inter_set)
        # case: the set has only one num
        if len(__list_inter) == 1:
            return  self.current_map.list_grids[__list_inter[0]-1].cost
        # case: the list has 2 nums 
        elif len(__list_inter) == 2:
            return np.round((self.current_map.list_grids[__list_inter[0]-1].cost \
                + self.current_map.list_grids[__list_inter[1]-1].cost) / 2, 1)
        else: 
            # invalid set
            print(f'[Error] Not Found intersection grids!')
            return None

    def is_accessiable(self, current_point, other_point):
        '''
        The method is to check if the edge is between 2 playing ground, then return False
        :param: current point
        :param: other point
        :return: True or false
        '''
        # get the intersection set of the 2 points' belong_num_grids
        __inter_list = list(current_point.belong_num_grids.intersection(other_point.belong_num_grids))
        if len(__inter_list) == 2:
            # check the accessiable
            if self.current_map.list_grids[__inter_list[0] - 1].type == "place_p" and \
                self.current_map.list_grids[__inter_list[1] - 1].type == "place_p":
                print(f'[Edge Not Access] Point {current_point.x, current_point.y} to {other_point.x, other_point.y} !')
                return False
        return True

    def is_quarantine_area(self, current_point):
        '''
        The method is to check if the point is in the quarantine area or not
        :param: current point
        :return: True or False
        '''
        # get index from the list_points
        __current_index = self.__get_num_point_in_list(current_point.x, current_point.y)
        # check if the current_point's belong_num_grids has the quarantine type
        for g_i in  self.current_map.list_points[__current_index].belong_num_grids:
            if self.current_map.list_grids[g_i - 1].type == "place_q":
                return True
        # noice for end point
        if current_point == self.end_point:
            print(f'[Error] End Point {current_point.x, current_point.y} is not in the quarantine area!')
        return False

    def __get_num_point_in_list(self, current_point_x, current_point_y):
        '''
        The method is to get the num of the point in the myMap.list_points
        :param: current point x, y
        :return: index : the num of the point in the myMap.list_points
        '''
        for index, p_i in enumerate(self.current_map.list_points):
            if current_point_x == p_i.x and current_point_y == p_i.y:
                return index
        return None

    def __push_open_list(self, current_point, other_x, other_y):
        '''
        The method is to calculating the g_cost of 4 directions neighbors if exists
        then push in the openlist
        :param: current point
        :param: other point's x, y
        '''
        __num = self.__get_num_point_in_list(other_x, other_y)
        if __num != None:
            # print(f'[Debug] Point num is : {__num}')
            __c_index = self.__get_num_point_in_list(current_point.x, current_point.y)
            __cost = self.calc_cost_with_2_points(current_point, self.current_map.list_points[__num])
             
            # if the edge is not accessiable, set the g_cost is infinum
            if self.is_accessiable(self.current_map.list_points[__c_index], self.current_map.list_points[__num]) and \
                    __cost != None: 
                # if the other point already is in the open list, then compare its f_cost,
                # if less than, update its g,f cost and parent.
                if self.__is_in_open_list(self.current_map.list_points[__num]):
                    # calculate f cost
                    __point_g_cost = __cost + self.current_map.list_points[__c_index].g_cost
                    __point_h_cost = self.__calc_h_cost(self.current_map.list_points[__num])
                    __point_f_cost = __point_g_cost + __point_h_cost
                    # copare the f_cost of the point which in the openList
                    if __point_f_cost < self.current_map.list_points[__num].f_cost:
                        print(f'[Update OpenList] The new f_cost {__point_f_cost} ([old f_cost is {self.current_map.list_points[__num].f_cost}]) of Point {self.current_map.list_points[__num].x, self.current_map.list_points[__num].y} ')
                        # update the info of the point
                        self.current_map.list_points[__num].parent = self.current_map.list_points[__c_index]
                        self.current_map.list_points[__num].g_cost = __point_g_cost
                        self.current_map.list_points[__num].h_cost = __point_h_cost
                        self.current_map.list_points[__num].f_cost = __point_f_cost
                        # test
                        print(f'   [Point] from {current_point.x, current_point.y} to {other_x, other_y}')
                        print(f'   [h cost] is {self.current_map.list_points[__num].h_cost}')
                        print(f'   [g cost] is {self.current_map.list_points[__num].g_cost}' )
                        print(f'   [f cost] is {self.current_map.list_points[__num].f_cost} \n' )
                    
                # if not in the close and open list, and g_cost < INFINUM, then push into the open list
                if not self.__is_in_close_list(self.current_map.list_points[__num]) \
                    and not self.__is_in_open_list(self.current_map.list_points[__num]) \
                        and (self.current_map.list_points[__num].g_cost < self._INFINUM):    
                    # assign the parent and g_cost of the next point
                    self.current_map.list_points[__num].parent = self.current_map.list_points[__c_index]        
                    # print(f'   [Debug] parent point is {self.current_map.list_points[__num].parent.x,self.current_map.list_points[__num].parent.y}')    
                    self.current_map.list_points[__num].g_cost = __cost + self.current_map.list_points[__num].parent.g_cost
                    # calculate h_cost
                    self.current_map.list_points[__num].h_cost = self.__calc_h_cost(self.current_map.list_points[__num])
                    # calculate the f_cost
                    self.current_map.list_points[__num].f_cost = self.current_map.list_points[__num].h_cost + self.current_map.list_points[__num].g_cost
                       
                    print(f'   [Point] from {current_point.x, current_point.y} to {other_x, other_y}')
                    print(f'   [h cost] is {self.current_map.list_points[__num].h_cost}')
                    print(f'   [g cost] is {self.current_map.list_points[__num].g_cost}' )
                    print(f'   [f cost] is {self.current_map.list_points[__num].f_cost} \n' )
                             
                    # heappush into the open list
                    heappush(self.open_list,self.current_map.list_points[__num])
                    
            else:
                self.current_map.list_points[__num].g_cost = self._INFINUM
        else: 
            # print("[Debug] Cannot find the point! \n")
            return
            
    def __calc_h_cost(self, current_point):
        '''
        calculate heuristic cost
        :param: current_point
        :return: h_cost of the current node
        --- Use Manhattan distance: abs(point.x - goal.x) + abs(point.y - goal.y)
        '''
        # since our width * height are 0.2*0.1, I decide multiply the result by 10 (eg. 0.1*10=1)
        # Also, the (point.x - goal.x) value needs to divides 2, to let the cell becomes square. 
        # Thus, it will make sure the steps of the points are equal in 4 different directions.
        __h_cost = np.round(10*(abs((current_point.x - self.goal_point.x) / 2) + abs(current_point.y - self.goal_point.y)),1)
        return __h_cost

    def __is_in_close_list(self, current_point):
        '''
        The method is to check if the point is in the close list or not
        :param: current point
        :return: True or False
        '''
        for p_i in self.close_list:
            if current_point == p_i:
                return True
        return False

    def __is_in_open_list(self, current_point):
        '''
        The method is to check if the point is in the open list or not
        :param: current point
        :return: True or False
        '''
        for p_i in self.open_list:
            if current_point == p_i:
                return True
        return False
    

    def __get_path(self):
        '''
        The method is to return the path and display the path and the total f cost
        '''
        print("*"*50)
        # case: arrive in the end point
        if len(self.close_list) > 0:       
            print("[Found Path] Path list is : ")
            # from end node to traverse path to the start node, then store in the path list
            
            # first, add 'end' node (sometimes is not true end point called arrive point)
            self.path_list.append(self.close_list[-1])
            # traverse all parent node of the arrive point
            # print(f'[Debug] arrive point is {self.close_list[-1].x, self.close_list[-1].y}')
            # print(f'[Debug1] parent point is {self.close_list[-1].parent.x, self.close_list[-1].parent.y}')
            # print(f'[Debug] start point is {self.start_point.x, self.start_point.y}')
            for i in range(len(self.close_list)):
                if self.path_list[i] == self.start_point:
                    break
                else:
                    self.path_list.append(self.path_list[-1].parent)
                    # print(f'[Debug] parent point is {self.path_list[-1].x, self.path_list[-1].y}')
                

            # reverse path list
            self.path_list.reverse()
            # f_cost needs to minus h_cost of the arrive point, since we need to set h_cost of arrive point to 0.
            self.path_list[-1].f_cost = self.path_list[-1].f_cost - self.path_list[-1].h_cost
            # display the total f cost, only need to print the last node's f_cost of the close_list    
            print(f'[Final f cost] is : {self.path_list[-1].f_cost}')       
            # print path list
            for item in self.path_list:
                print(f'{item.x, item.y}', end = ' ')
            # store in a PathNode 
            print("", end='\n')
            print("[Storing] the found path in the path_list!")
            self.create_path_node(self.path_list, self.path_list[-1].f_cost)
        else:
            print("[Not Found Path] Close list is empty!")
        print("*"*50)
    
    def create_path_node(self, found_path, f_cost):
        '''
        The method is to create a path node to store in the path_list.
        :param: found path : a found path 
        '''
        # using numpy to store the path list
        __temp_path = []
        for item in found_path:
            __temp_path.append(item.x)
            __temp_path.append(item.y)
        # for draw path in matplotlib, a narray with 2 columns
        __found_path = np.array(__temp_path).reshape(-1,2)
        # using heap to store the PathNode
        heappush(self.path_nodes_list, self.PathNode(__found_path, f_cost))

    def create_goal_points_list(self):
        '''
        The method is to create a list to store goal points.
        follow by the numbers of the points in the quarantine cells,
        store them in the goal_points_list
        '''
        # if the point is in a quarantine area, then store it as a goal point
        __temp_num = 0
        for i_p in range(len(self.current_map.list_points)):
            if self.is_quarantine_area(self.current_map.list_points[i_p]):
                self.goal_points_list.append(self.current_map.list_points[i_p])
                __temp_num += 1
        print(f'[News] Set total {__temp_num} goal points to search!')   
        print("-"*50)

    def run_search(self,goal_point):
        '''
        The method is to run A star search
        '''  
        # (4) check the end point is in the quarantine place
        if self.is_quarantine_area(goal_point):
            print("-"*50)
            print(f'[News] Valid goal point {goal_point.x, goal_point.y}, searching...')
            # heapify the openlist
            heapify(self.open_list)
            # start search: using heapq for implement priprity queue 
            __p_index = self.__get_num_point_in_list(self.start_point.x, self.start_point.y)
            heappush(self.open_list, self.current_map.list_points[__p_index])
            while self.open_list:
                # pop a point with smallest f_cost from the open list
                current_node = heappop(self.open_list)
                # add to close list 
                self.close_list.append(current_node)
                # check if current node reached to the goal node
                # or the current node is already in the quarantine area
                if current_node == goal_point or self.is_quarantine_area(current_node):
                    # Recall fuction to return "reverse" close list as path list
                    print("[News] Found the path!") 
                    self.__get_path()
                    break 
                # calculating the g_cost of 4 directions neighbors if exists
                # then push in the openlist
                print("-"*50)
                # order is 'up, down, left, right' respectively
                print("[Searching] 4 directions ('up, down, left, right'), if the point is in the close list, then not search!")
                self.__push_open_list(current_node, current_node.x, np.round(current_node.y + self.current_map.offsetY,1))
                self.__push_open_list(current_node, current_node.x, np.round(current_node.y - self.current_map.offsetY,1))
                self.__push_open_list(current_node, np.round(current_node.x - self.current_map.offsetX,1), current_node.y)
                self.__push_open_list(current_node, np.round(current_node.x + self.current_map.offsetX,1), current_node.y)
                print("[Completed] Have searched 4 directions!")
                print("="*50)
        

    def start(self):
        '''
        The method is to start the A* algorithm search
        '''
        # print Role C or V
        print(f'[Role] The current Role is {self.current_map.role_type}\n')
        # find all correct goal points
        self.create_goal_points_list()

        # (1) check the start and end point are valid
        if self.is_valid_point(self.start_point) and self.is_valid_point(self.end_point):
            # continue A* search
            print("[News] Valid points, A star searching...")
            # (2) transform start and end point
            self.transform_point(self.start_point)
            self.transform_point(self.end_point)
            print(f'[Transform] Start Point {self.start_point.x, self.start_point.y}, End Point {self.end_point.x, self.end_point.y}')
            # (3) check the start point is already in the quarantine place
            if self.is_quarantine_area(self.start_point):
                print(f'[No Path Found] The start point {self.start_point.x, self.start_point.y} is already in the quarantine place!')
                return
            print("[News] Valid Start points, searching...")

            # heapify the path_nodes_list
            heapify(self.path_nodes_list)

            # set goal point in order
            for goal_point_item in self.goal_points_list:
                # print(f'test {goal_point_item.x,goal_point_item.y }')
                # set goal_point for calculate h_cost
                self.goal_point = goal_point_item
                # run A* search
                self.run_search(goal_point_item)
                # reset search conditions
                self.__reset_search()
            # finish the search
            print("[Finished Search] Finding the optimal path...")
            # find the optimal path
            self.__find_optimal_path()

        else:
            print("[Error] invalid input, please check again!")

    def __find_optimal_path(self):
        '''
        The method is to find the optimal path.
        '''
        # using priority queue to get the optimal PathNode with the smallest total cost
        self.optimal_path_node = heappop(self.path_nodes_list)
        self.optimal_path = self.optimal_path_node.found_path
        self.optimal_path_total_cost = self.optimal_path_node.total_cost
        print("-"*50)
        print("[Optimal Path]")
        print(f'{self.optimal_path}')
        print(f'[Total Cost] {self.optimal_path_total_cost}')
        print("-"*50)

    def get_optimal_path(self):
        '''
        The method is to return the optimal path.
        :return: optimal_path
        '''
        return self.optimal_path

    def __reset_search(self):
        '''
        The method is to reset search conditions.
        such as the h,g,f costs of the points, clear open_list, close_list, etc
        '''
        # reset the h,g,f costs of the points to 0.0
        for i in range(len(self.current_map.list_points)):
            self.current_map.list_points[i].h_cost = 0.0
            self.current_map.list_points[i].g_cost = 0.0
            self.current_map.list_points[i].f_cost = 0.0
            self.current_map.list_points[i].parent = None
        # clear open and close list
        self.open_list.clear()
        self.close_list.clear()
        self.path_list.clear()
            




   

        


        


                




        

        
        
