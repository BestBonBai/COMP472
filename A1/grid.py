# -------------------------------------------------------
# Assignment (1)
# Written by (Hualin Bai : 40053833)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------


class Grid:
    '''
    Class grid: store each grid's info

    @ different types: for fill different color in map
        place_q(color:green): quarantine place, 
        place_v(color: yellow):vaccine spot, 
        place_p(color: blue): playing ground

    @ bottom point: for draw rectangle in map

    '''
    __type = "number" # private type of grid
    __num = 0 # private num of each grid
    __row = 0 # private row num
    __column = 0 # private column num
    # private variables for 4 directions neighbors, default = -1 means undefined
    __grid_up_num, __grid_down_num, __grid_left_num, __grid_right_num = -1,-1,-1,-1
    __lst_neighbors = [] # for neighbors list
    __lst_grids = [] # for list grids

    def __init__(self, x, y, num, r_num, c_num):
        '''
        x, y are bottom-point of each grid for draw rectangle in map
        num is the number of each grid
        row and column are to find all neighbors of the grid
        '''
        self.x = x
        self.y = y
        self.__num = num
        self.__row = r_num
        self.__column = c_num
    
    def set_type(self, type):
        if(type == "place_q"):
            self.__type = "place_q"
        elif(type == "place_v"):
            self.__type = "place_v"
        elif(type == "place_p"):
            self.__type = "place_p"
        else: 
            print("invalid type!!! Using default type: number!!!")
    
    def get_type(self):
        """ return the type of the specific grid """
        return self.__type

    def get_bot_ptr(self):
        """ return the bottom-point of the specific grid """    
        return self.x, self.y

    def get_top_right_point(self, offsetX, offsetY):
        """ return the most top right point of the grid """
        return self.x + offsetX, self.y + offsetY    

    def get_grid_num(self):
        """ return the number of the grid """
        return self.__num

    def __find_neighbors_num(self):
        """ Private method: find the neighbors' num of the grid if it exists"""
        # find up, bottom, left, right neighbours
        self.__lst_neighbors = []
        self.size = self.__row * self.__column
        # (1) for up 
        if(self.__num - self.__column > 0 and self.__num - self.__column <= self.size):
            self.__grid_up_num = self.__num - self.__column
            self.__lst_neighbors.append(self.__grid_up_num)
        # (2) for down
        if(self.__num + self.__column > 0 and self.__num + self.__column <= self.size ):
            self.__grid_down_num = self.__num + self.__column   
            self.__lst_neighbors.append(self.__grid_down_num)
        # (3) for left
        if(self.__num % self.__column != 1):
            if( self.__num - 1 > 0 and self.__num - 1 <= self.__row * self.__column ):
                self.__grid_left_num = self.__num - 1 
                self.__lst_neighbors.append(self.__grid_left_num)
        # (4) for right  
        if(self.__num % self.__column != 0):  
            if(self.__num + 1 > 0 and self.__num + 1 <= self.__row * self.__column ):
                self.__grid_right_num = self.__num + 1 
                self.__lst_neighbors.append(self.__grid_right_num)


    def get_neighbors_num(self):
        """ Return 4 directions neighbors numbers if exists
        Using __find_neighbors_num """
        self.__find_neighbors_num() 

        # check if neighbor exits
        if(len(self.__lst_neighbors) > 0):
            return self.__lst_neighbors   
        else:
            print("Not found neighbors")