## Assignment 1

### Required Environment
- python 3.7
- conda 
    - `conda active py37`

### lists of libaries
- matplotlib
- numpy
- sys
- math (for module calculation)
- heapq (for priority queue)

### File Structures
```
|---[A1]
    |-----A1_test.ipynb
    |-----draw.py
    |-----create_map.py
    |-----a_star_2_roles.py
    |-----[images]
        |-------original_map.jpg
        |-------optimal_path_map_role_c.jpg
        |-------optimal_path_map_role_v.jpg
    |-----Expectation-of-Originality-40053833.pdf
    |-----Expectation-of-Originality-40079344.pdf
```

### Run Instructions:

#### first method:
1. cd into `A1` folder
2. run `draw.py` using `python3 draw.py`
3. following the input requirements in the terminal
    - input row and column num
    - input number to choose role type (C or V) (1 for Role C, 2 for Role V)
    - input numbers of the grid cells to assign different types, separated by comma (eg. 1,2,3)
        - assign for the quarantine place, the vaccine spot, the playing ground, separately.
    - input (x,y) coordinates of start and end points, separated by comma (eg. 0.2,0.3)
    - input y/n to decide whether start to A star search
    - input y/n to decide whether to draw path 
4. check the map and path images in `images` folder
#### second method:
1. cd into `A1` folder
2. run `jupyter notebook`
3. using `jupyter notebook` to open `A1_test.ipynb`
4. run each `block` in order
5. check the result