# -------------------------------------------------------
# Assignment (1)
# Written by (Hualin Bai : 40053833 , Jiawei Zeng : 40079344)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------

import numpy as np

class MyMap:
    '''
    Class map: store the info of the map
    :param: row
    :param: column
    @ xlim
    @ ylim
    @ offsetX : for width * length = 0.1 * 0.2
    @ offsetY
    @ list_grids: for store all grids
    @ list_points: for store all points

    '''
    def __init__(self, row, column):
        '''
        Inital the class myMap
        :param: row
        :param: column
        '''
        self.row = row
        self.column = column
        self.offsetX = 0.2
        self.offsetY = 0.1
        self.xlim = np.round(self.column * self.offsetX,1)
        self.ylim = np.round(self.row * self.offsetY,1)
        self.list_grids = [] 
        self.list_points = []
        # type of role
        self.role_type = 'role_c' # default is role c
    
    def get_axis_scales(self):
        '''
        The method is to set axis scales
        :return: x_scales
        :return: y_scales
        '''
        # create the scales of the axis
        self.x_scales = np.round(np.linspace(0, self.xlim, self.column + 1),1)
        self.y_scales = np.round(np.linspace(0, self.ylim, self.row + 1),1)
        return self.x_scales, self.y_scales


    def create_all_grids(self):
        '''
        The method is to create each grid cell in map.
        '''
        # set each grid's bottom-point for draw Rectangle
        self.num_grids = self.row * self.column
        self.x_rect_ptr_arr = np.round(self.x_scales[0:-1],1) # keep 0.1f
        self.y_rect_ptr_arr = np.round(self.y_scales[0:-1],1)
        # reverse y_rect_ptr_arr
        self.y_rect_ptr_arr = np.flip(self.y_rect_ptr_arr)
        
        n=1
        # assign each grid's bottom-point
        for yi in self.y_rect_ptr_arr:
            for xi in self.x_rect_ptr_arr:
                self.list_grids.append(Grid(xi,yi,n, self.offsetX,self.offsetY, self.role_type))
                n += 1
        print(f'[news] Total Num of Grids is: {len(self.list_grids)} ')

    def create_all_points(self):
        '''
        The method is to create all intersection points in the map.
        '''
        # assign each points
        for p_xi in self.x_scales:
            for p_yi in self.y_scales:
                self.list_points.append(Point(p_xi,p_yi))
        print(f'[news] Total num of created points is : {len(self.list_points)}')
        # set the belong num of grids for each points
        for p_i in self.list_points:
            for grid_i in self.list_grids:
                # check if the point is the same as 4 points of the grid,
                # then add the num of grid into its belong_num_grids(set type)
                __x1, __y1 = grid_i.get_bottom_left_point()
                __x2, __y2 = grid_i.get_top_right_point()
                __x3, __y3 = grid_i.get_top_left_point()
                __x4, __y4 = grid_i.get_bottom_right_point()
                if (p_i.x == __x1 and p_i.y == __y1) or (p_i.x == __x2 and p_i.y == __y2) \
                    or (p_i.x == __x3 and p_i.y == __y3) or (p_i.x == __x4 and p_i.y == __y4):
                    p_i.belong_num_grids.add(grid_i.num)
        print(f'[news] already set the belong grids num for all points!\n')

            
        
        


class Grid:
    '''
    Class grid: store each grid's info

    @ different types: for fill different color in map
        place_q(color:green): quarantine place, 
        place_v(color: yellow):vaccine spot, 
        place_p(color: blue): playing ground

    @ left-bottom point: for draw rectangle in map
    @ right-top point: top_right_x, top_right_y,
    @dict_neighbors: return a dictionary with 4 directions neighbors
    :get_neighbors_num

    @cost: (Role C) actual cost of the grid cell defined in A1 for calculate g(n) cost
    '''

    def __init__(self, x, y, num, map_offsetX, map_offsetY, role_type):
        '''
        :param: x, y : are bottom-point of each grid for draw rectangle in map
        :param: num is the number of each grid
        :param: map_xxx : the info of the map 
        '''
        self.x = x # the bottom-left-point of the grid
        self.y = y
        self.num = num # the num of the grid
        self.offsetX = map_offsetX
        self.offsetY = map_offsetY
        self.type = "number" # type of the grid
        self.role_type = role_type
        if role_type == 'role_c':
            # set default number grid is 1.0 cost
            self.cost = 1.0
        elif role_type == 'role_v':
            # set default number grid is 2.0 cost
            self.cost = 2.0
        else:
            print("[Debug] Invalid Role Type! Check Again!")
        # initial the min and max of the x and y
        self.x_min = self.x
        self.x_max = np.round(self.x + self.offsetX,1)
        self.y_min = self.y
        self.y_max = np.round(self.y + self.offsetY,1)
    
    def set_type(self, type):
        # check role type
        if self.role_type == 'role_c':
            if(type == "place_q"):
                self.type = "place_q"
                self.cost = 0.0
            elif(type == "place_v"):
                self.type = "place_v"
                self.cost = 2.0
            elif(type == "place_p"):
                self.type = "place_p"
                self.cost = 3.0
            else: 
                self.cost = 1.0
                print("[Debug] invalid type!!! Using default type: number!!!")
        if self.role_type == 'role_v':
            if(type == "place_q"):
                self.type = "place_q"
                self.cost = 3.0
            elif(type == "place_v"):
                self.type = "place_v"
                self.cost = 0.0
            elif(type == "place_p"):
                self.type = "place_p"
                self.cost = 1.0
            else: 
                self.cost = 2.0
                print("[Debug] invalid type!!! Using default type: number!!!")


    def get_bottom_left_point(self):
        """ return the bottom-left-point of the specific grid """    
        return self.x, self.y

    def get_top_right_point(self):
        """ return the most top right point of the grid """
        self.top_right_x = np.round(self.x + self.offsetX, 1)
        self.top_right_y = np.round(self.y + self.offsetY, 1)
        return self.top_right_x, self.top_right_y

    def get_top_left_point(self):
        """ return the most top left point of the grid """
        self.top_left_x = np.round(self.x, 1)
        self.top_left_y = np.round(self.y + self.offsetY, 1)
        return self.top_left_x, self.top_left_y

    def get_bottom_right_point(self):
        """ return the most bottom right point of the grid """
        self.bottom_right_x = np.round(self.x + self.offsetX, 1)
        self.bottom_right_y = np.round(self.y, 1)
        return self.bottom_right_x, self.bottom_right_y  
        



class Point:
    '''
    The class is to store the info of the point in the map.
    :param: x, y : are coordinates in the map
    '''  
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.belong_num_grids = set() # a set of belong num of grids
        # initial costs
        self.g_cost = 0.0
        self.h_cost = 0.0
        self.f_cost = 0.0
        self.parent = None

    def __eq__(self, other):
        '''
        @ override the equal method
        :return: Ture or False
        '''
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __lt__(self,other):
        '''
        @ override compare < method
        compare the f_cost of 2 Points
        :return: True or False
        '''
        if self.f_cost < other.f_cost:
            return True
        return False
