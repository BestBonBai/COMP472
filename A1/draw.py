# -------------------------------------------------------
# Assignment (1)
# Written by (Hualin Bai : 40053833 , Jiawei Zeng : 40079344)
# For COMP 472 Section (LAB AI-X) – Summer 2021
# --------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from create_map import MyMap, Grid, Point
from a_star_2_roles import A_Star_Search



def main():
    print("-"*50)
    print("Covid-19 Map Simulation")
    print("Generating map...")
    print("-"*50)

    fig, ax= plt.subplots(figsize=(10,5)) # set a 5X10 Canvas

    while True:
        try:
            # Row & Column need to be inputted by user
            row = int(input("Enter row num of the map (type: int, range > 0) :"))
            column = int(input("Enter column num of the map (type: int, range > 0) :"))
        except ValueError:
            print("[Error] invalid Row or Column value! Please input again!")
        else:
            if isinstance(row, int) and row > 0 and isinstance(column, int) and column > 0:
                print("[News] Valid Row and Column value")
                break
            else:
                print("[Error] invalid Row or Column value! Please input again!")
            

    # create a new map
    aMap = MyMap(row, column)

    # set the axis range
    ax.set_xlim(0,aMap.xlim)
    ax.set_ylim(0,aMap.ylim)

    # create the scales of the axis
    x_arr, y_arr = aMap.get_axis_scales()
    # x_arr = np.linspace(0, aMap.xlim, column+1)
    # y_arr = np.linspace(0, aMap.ylim, row+1)

    # test the values of X and Y axis
    print(x_arr)
    print(y_arr)

    #set axis scales
    ax.set_xticks(x_arr)
    ax.set_yticks(y_arr)

    # set Role type: role C or role V to set costs in grids
    while True:
        try: 
            # choose Role
            role_type = int(input(f'Please Choose a Role to search (1 for Role C, 2 for Role V) : '))
        except ValueError:
            print("[Error] Invalid Role Type! Check Again!")
        else:
            # set role
            if role_type == 1:
                aMap.role_type = 'role_c'
                print("[News] You have set Role C \n")
                break
            elif role_type == 2:
                aMap.role_type = 'role_v'
                print("[News] You have set Role V \n")
                break
            else:
                print("[Error] Invalid Role Type! Check Again!")

    # set title and axis-lables
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("COVID-19 Map: A* Search Algorithm", fontsize=16, color='black', verticalalignment="baseline")   

    # create all grids
    aMap.create_all_grids()

    while True:
        try:
            # user input different types of the grids
            q_input = input(f'Enter numbers of the quarantine place, separated by comma (eg. 1,2,3) [range is from 1 to {row*column}]:').split(",")  
            # check the number is valid
            if all(1 <= int(q) <= (row*column) for q in q_input):
                q_arr = [int(q_input[i]) for i in range(len(q_input))]
                break
            else: print("[Error] Invalid Input numbers of place, please Check again!")
        except ValueError:
            print("[Error] Invalid Input numbers of place, please Check again!")
    
    while True:
        try:
            # user input different types of the grids
            v_input = input(f'Enter numbers of the vaccine spot, separated by comma (eg. 1,2,3) [range is from 1 to {row*column}]:').split(",")            
            # check the number is valid
            if all(1 <= int(v) <= (row*column) for v in v_input):
                v_arr = [int(v_input[i]) for i in range(len(v_input))]
                break
            else: print("[Error] Invalid Input numbers of place, please Check again!")
        except ValueError:
            print("[Error] Invalid Input numbers of place, please Check again!")

    while True:
        try:
            # user input different types of the grids
            p_input = input(f'Enter numbers of the playing ground, separated by comma (eg. 1,2,3) [range is from 1 to {row*column}]:').split(",")         
            
            # check the number is valid
            if all(1 <= int(p) <= (row*column) for p in p_input):
                p_arr = [int(p_input[i]) for i in range(len(p_input))]
                break
            else:
                print("[Error] Invalid Input numbers of place, please Check again!")
        except ValueError:
            print("[Error] Invalid Input numbers of place, please Check again!")






    # set each grid's type
    for qi in q_arr:
        aMap.list_grids[qi-1].set_type("place_q")
        
    for vi in v_arr:
        aMap.list_grids[vi-1].set_type("place_v")
        
    for pi in p_arr:
        aMap.list_grids[pi-1].set_type("place_p")  

    # DEBUG Use 
    # for item in aMap.list_grids:
        # print( "Num [" + str(item.num) + "] is " + str(item.type))   
        # print("Bottom-left point is : " + str(item.get_bottom_left_point()))
        # print("Top-right point is : " + str(item.get_top_right_point()) + "\n")

    # set 3 types for each grid by user: 
    # place_q(color:green): quarantine place, place_v(color: orange):vaccine spot, place_p(color: blue): playing ground

    #set different color for each grid
    # use Rectangle to draw each grid, xy is the bottom-left point of each grid
    for item in aMap.list_grids:
        if item.type == "place_q":
            rect = mpatches.Rectangle(item.get_bottom_left_point(), aMap.offsetX, aMap.offsetY, edgecolor="black", facecolor="g")
            ax.add_patch(rect)
        elif item.type == "place_v":
            rect = mpatches.Rectangle(item.get_bottom_left_point(), aMap.offsetX, aMap.offsetY, edgecolor="black", facecolor="orange")
            ax.add_patch(rect)
        elif item.type == "place_p":
            rect = mpatches.Rectangle(item.get_bottom_left_point(), aMap.offsetX, aMap.offsetY, edgecolor="black", facecolor="r")
            ax.add_patch(rect)
        else: 
            pass
        # add text in map for grid's number
        ax.text(item.x + aMap.offsetX/2,item.y+aMap.offsetY/2,str(item.num), fontsize=12, color="k", alpha=1)
        
        
    # set legend
    patch_q = mpatches.Patch(color='green', label='quarantine place')
    patch_v = mpatches.Patch(color='orange', label='vaccine spot')
    patch_p = mpatches.Patch(color='red', label='playing ground')
    plt.legend(handles=[patch_q, patch_v, patch_p], bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()


    ax.grid(True)
    # open to display the map, or comment for continue type command in the terminal
    # plt.show()
    
    # create all intersection points in the map
    aMap.create_all_points()
    # test each points's belong num grids
    # for item in aMap.list_points:
    #     print(f'{item.x, item.y} Belong num grids are : {item.belong_num_grids}')
    
    # save figure image (check in the folder named images)
    fig.savefig(r'./images/original_map.jpg')
    print("\n[Image] The map has saved in the image folder! \n[Help] Please Open (original_map.jpg) to Choose start/end point!\n")

    

    # set start and end point
    while True:
        try:
            # set start and end point
            start_x, start_y = map(float, input(f'Enter start point\'s x between [0, {aMap.xlim}] and y between [0,{aMap.ylim}], separated by comma (eg. 0.2,0.3)').split(",")) 
            end_x, end_y = map(float, input(f'Enter end point\'s x between [0, {aMap.xlim}] and y between [0,{aMap.ylim}], separated by comma (eg. 0.2,0.3)').split(",")) 
        except ValueError:
            print("[Error] Invalid start or end point! Check Again!")
        else:
            if start_x >= 0 and start_x <= aMap.xlim and start_y >=0 and start_y <= aMap.ylim \
                and end_x >= 0 and end_x <= aMap.xlim and end_y >=0 and end_y <= aMap.ylim :
                print("[News] Valid Points in the Map, continue to checking...!")
                break
            else:
                print("[Error] Invalid start or end point! Input again!")
            
    
    # if valid input start and end point, then create Point instance for start and end point
    start_point = Point(start_x, start_y)
    end_point = Point(end_x, end_y) 
    # start search
    while True:
        draw_path_input = input("\nEnter y/n to start A star search: (y/n)")
        if draw_path_input == 'y':
            # run A star search
            myAstar = A_Star_Search(start_point,end_point, aMap)
            myAstar.start()
            break
        elif draw_path_input == 'n':
            sys.exit(0)
        else:
            print("\n[Warning] Invalid Input! Please Check Again!")
    
    # display the path
    while True:
        draw_path_input = input("\nEnter y/n to display the path or not: (y/n)")
        if draw_path_input == 'y':
            break
        elif draw_path_input == 'n':
            sys.exit(0)
        else:
            print("\n[Warning] Invalid Input! Please Check Again!")
        
    # check if the path exists
    if myAstar.has_path:
        # draw Path
        myPath = myAstar.get_optimal_path()
        print(myPath)
        # draw path
        ax.plot(*zip(*myPath), marker='o', color = 'k', linestyle='dashed', linewidth=3, markersize=12, label='path')

        # draw start and end point's colors
        ax.plot(*zip(myPath[0]),marker='o', color = 'b', markersize=12)
        ax.plot(*zip(myPath[-1]),marker='o', color = 'm', markersize=12)

        # update the legend lables
        patch_path = mpatches.Patch(color='black', label='found path')
        patch_start = mpatches.Patch(color='blue', label='start point')
        patch_end = mpatches.Patch(color='m', label='arrive point')
        ax.legend(handles=[patch_q, patch_v, patch_p, patch_path, patch_start, patch_end], bbox_to_anchor=(1.05, 1.0), loc='upper left')

        if aMap.role_type == 'role_c':
            # change title
            ax.set_title("COVID-19 Map: A* Search Algorithm (Role C)", fontsize=16, color='black', verticalalignment="baseline")
            # display the figure
            # save figure image (check in the folder named images)
            fig.savefig(r'./images/optimal_path_map_role_c.jpg')
            print("\n[Image] The path of the map has saved in the image folder!")
            print("Please Check (optimal_path_map_role_c.jpg) in the image folder!\n")
        if aMap.role_type == 'role_v':
            # change title
            ax.set_title("COVID-19 Map: A* Search Algorithm (Role V)", fontsize=16, color='black', verticalalignment="baseline")
            # display the figure
            # save figure image (check in the folder named images)
            fig.savefig(r'./images/optimal_path_map_role_v.jpg')
            print("\n[Image] The path of the map has saved in the image folder!")
            print("Please Check (optimal_path_map_role_v.jpg) in the image folder!\n")
        print("Please Close the window of the Matplot to terminate the program!")
        # fig
        plt.show()
    else:
        print("[No Path is Found] Please Try Again!\n")
        sys.exit(0)
    




if __name__ == '__main__':
    main()
    