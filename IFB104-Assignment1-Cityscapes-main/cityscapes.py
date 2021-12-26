
#-----Statement of Authorship----------------------------------------#
#
#  This is an individual assessment item.  By submitting this
#  code I agree that it represents my own work.  I am aware of
#  the University rule that a student must not act in a manner
#  which constitutes academic dishonesty as stated and explained
#  in QUT's Manual of Policies and Procedures, Section C/5.3
#  "Academic Integrity" and Section E/2.1 "Student Code of Conduct".
#
#
#  NB: Files submitted without a completed copy of this statement
#  will not be marked.  All files submitted will be subjected to
#  software plagiarism analysis using the MoSS system
#  (http://theory.stanford.edu/~aiken/moss/).
#
#--------------------------------------------------------------------#



#-----Assignment Description---------------------5--------------------#
#
#  CITYSCAPES
#
#  This assignment tests your skills at defining functions, processing
#  data stored in lists and efficiently repeating multiple actions in
#  order to display a complex visual image.  The incomplete
#  Python script below is missing a crucial function, "build_city".
#  You are required to complete this function so that when the
#  program is run it draws a city whose plan is determined by
#  randomly-generated data stored in a list which specifies what
#  style of building to erect on particular sites.  See the
#  instruction sheet accompanying this file for full details.
#
#  Note that this assignment is in two parts, the second of which
#  will be released only just before the final deadline.  This
#  template file will be used for both parts and you will submit
#  your final solution as a single Python 3 file, whether or not you
#  complete both parts of the assignment.
#
#--------------------------------------------------------------------#  



#-----Preamble-------------------------------------------------------#
#
# This section imports necessary functions and defines constant
# values used for creating the drawing canvas.  You should not change
# any of the code in this section.
#

# Import the functions needed to complete this assignment.  You
# should not need to use any other modules for your solution.  In
# particular, your solution must not rely on any non-standard Python
# modules that need to be installed separately, because the markers
# may not have access to such modules.

from turtle import *
from math import *

# Define constant values used in the main program that sets up
# the drawing canvas.  Do not change any of these values.

canvas_height = 700 # pixels
canvas_width = 1100 # pixels
grass_depth = 95 # vertical depth of the "grass", in pixels
half_width = canvas_width // 2 # maximum x coordinate in either direction
grid_font = ('Arial', 10, 'normal') # font for drawing the grid
grid_size = 50 # gradations for the x and y scales shown on the screen
offset = 5 # offset of the x-y coordinates from the screen's edge, in pixels
max_height = canvas_height - grass_depth # maximum positive y coordinate
max_building_height = 575 # city ordinance maximum building height
site_width = 240 # maximum width of a building site

# Define the locations of building sites approved by the
# city council (arranged from back to front)
sites = [['Site 1', [-225, 0]],
         ['Site 2', [25, 0]],
         ['Site 3', [275, 0]],
         ['Site 4', [-375, -25]],
         ['Site 5', [-125, -25]],
         ['Site 6', [125, -25]],
         ['Site 7', [375, -25]],
         ['Site 8', [-275, -50]],
         ['Site 9', [-25, -50]],
         ['Site 10', [225, -50]]]

#
#--------------------------------------------------------------------#



#-----Functions for Creating the Drawing Canvas----------------------#
#
# The functions in this section are called by the main program to
# manage the drawing canvas for your image.  You should not change
# any of the code in this section.
#

# Set up the canvas and draw the background for the overall image.
# By default the drawing grid is displayed - call the function
# with False as the argument to prevent this.
def create_drawing_canvas(show_grid = False):

    # Set up the drawing canvas with coordinate (0, 0) in the
    # "grass" area
    setup(canvas_width, canvas_height)
    setworldcoordinates(-half_width, -grass_depth, half_width, max_height)

    # Draw as fast as possible
    tracer(False)

    # Make the sky blue
    bgcolor('sky blue')

    # Draw the "grass" as a big green rectangle (overlapping the
    # edge of the drawing canvas slightly)
    overlap = 25 # pixels
    penup()
    goto(-(half_width + overlap), -(grass_depth + overlap)) # bottom-left
    fillcolor('pale green')
    begin_fill()
    setheading(90) # face north
    forward(grass_depth + overlap * 2)
    right(90) # face east
    forward(canvas_width + overlap * 2)
    right(90) # face south
    forward(grass_depth + overlap * 2)
    end_fill()

    # Draw a nice warm sun peeking into the image at the top left
    penup()
    goto(-canvas_width // 2, canvas_height - grass_depth)
    pencolor('yellow')
    dot(350)

    # Draw a big fluffy white cloud in the sky
    goto(canvas_width // 3, canvas_height - grass_depth - 100)
    pencolor('white')
    dot(200)
    setheading(200)
    forward(100)
    dot(180)
    setheading(0)
    forward(200)
    dot(160)

    # Optionally draw x coordinates along the bottom of the
    # screen (to aid debugging and marking)
    pencolor('black')
    if show_grid:
        for x_coord in range(-half_width + grid_size, half_width, grid_size):
            goto(x_coord, -grass_depth + offset)
            write('| ' + str(x_coord), font = grid_font)

    # Optionally draw y coordinates on the left-hand edge of
    # the screen (to aid debugging and marking)
    if show_grid:
        for y_coord in range(-grid_size, max_height, grid_size):
            goto(-half_width + offset, y_coord - offset)
            write(y_coord, font = grid_font)
        goto(-half_width + offset, max_building_height - 5)
        write('Maximum allowed building height', font = grid_font)

    # Optionally mark each of the building sites approved by
    # the city council
    if show_grid:
        for site_name, location in sites:
            goto(location)
            dot(5)
            goto(location[0] - (site_width // 2), location[1])
            setheading(0)
            pendown()
            forward(site_width)
            penup()
            goto(location[0] - 40, location[1] - 17)
            write(site_name + ': ' + str(location), font = grid_font)
     
    # Reset everything ready for the student's solution
    pencolor('black')
    width(1)
    penup()
    home()
    tracer(True)


# End the program and release the drawing canvas.
# By default the cursor (turtle) is hidden when the program
# ends - call the function with False as the argument to
# prevent this.
def release_drawing_canvas(hide_cursor = True):
    tracer(True) # ensure any drawing in progress is displayed
    if hide_cursor:
        hideturtle()
    done()
    
#
#--------------------------------------------------------------------#



#-----Test Data for Use During Code Development----------------------#
#
# The "fixed" data sets in this section are provided to help you
# develop and test your code.  You can use them as the argument to
# the build_city function while perfecting your solution.  However,
# they will NOT be used to assess your program.  Your solution will
# be assessed using the random_plan function appearing below.  Your
# program must work correctly for any data set generated by the
# random_plan function.
#
# Each of the data sets below is a list specifying a set of
# buildings to be erected.  Each specification consists of the
# following parts:
#
# a) The site on which to erect the building, from Site 1 to 10.
# b) The style of building to be erected, from style 'A' to 'D'.
# c) The number of floors to be constructed, from 1 to 10.
# d) An extra value, either 'X' or 'O', whose purpose will be
#    revealed only in Part B of the assignment.  You should
#    ignore it while completing Part A.
#

# Each of these data sets draws just one building in each of the
# four styles
fixed_plan_1 = [[1, 'A', 6, 'O']]
fixed_plan_2 = [[2, 'B', 7, 'O']]
fixed_plan_3 = [[3, 'C', 5, 'O']]
fixed_plan_4 = [[4, 'D', 4, 'O']]
fixed_plan_5 = [[1, 'A', 9, 'X']]
fixed_plan_6 = [[2, 'B', 2, 'X']]
fixed_plan_7 = [[3, 'C', 3, 'X']]
fixed_plan_8 = [[4, 'D', 6, 'X']]

# Each of the following data sets draws just one style of
# building but at three different sizes, including the maximum
# (so that you can check your building's maximum height against
# the height limit imposed by the city council)
fixed_plan_9 = [[1, 'A', 10, 'O'], [2, 'A', 5, 'O'], [3, 'A', 1, 'O']]
fixed_plan_10 = [[1, 'B', 10, 'O'], [2, 'B', 5, 'O'], [3, 'B', 1, 'O']]
fixed_plan_11 = [[1, 'C', 10, 'O'], [2, 'C', 5, 'O'], [3, 'C', 1, 'O']]
fixed_plan_12 = [[1, 'D', 10, 'O'], [2, 'D', 5, 'O'], [3, 'D', 1, 'O']]
fixed_plan_13 = [[1, 'A', 10, 'X'], [2, 'A', 5, 'X'], [3, 'A', 1, 'X']]
fixed_plan_14 = [[1, 'B', 10, 'X'], [2, 'B', 5, 'X'], [3, 'B', 1, 'X']]
fixed_plan_15 = [[1, 'C', 10, 'X'], [2, 'C', 5, 'X'], [3, 'C', 1, 'X']]
fixed_plan_16 = [[1, 'D', 10, 'X'], [2, 'D', 5, 'X'], [3, 'D', 1, 'X']]

# Each of the following data sets draws a complete cityscape
# involving each style of building at least once. There is
# no pattern to them, they are simply specific examples of the
# kind of data returned by the random_plan function which will be
# used to assess your solution. Your program must work for any value
# that can be returned by the random_plan function, not just these
# fixed data sets.
fixed_plan_17 = \
         [[1, 'D', 2, 'O'],
          [2, 'B', 7, 'O'],
          [5, 'C', 6, 'O'],
          [6, 'A', 4, 'O']]
fixed_plan_18 = \
         [[1, 'D', 6, 'O'],
          [3, 'C', 5, 'O'],
          [4, 'B', 3, 'O'],
          [9, 'A', 9, 'O'],
          [10, 'D', 2, 'O']]
fixed_plan_19 = \
         [[5, 'C', 6, 'O'],
          [6, 'B', 9, 'O'],
          [7, 'A', 5, 'O'],
          [8, 'A', 7, 'O'],
          [9, 'D', 4, 'O']]
fixed_plan_20 = \
         [[1, 'A', 4, 'O'],
          [2, 'B', 4, 'O'],
          [3, 'A', 5, 'O'],
          [4, 'D', 7, 'O'],
          [10, 'B', 10, 'O']]
fixed_plan_21 = \
         [[1, 'B', 6, 'O'],
          [3, 'A', 4, 'O'],
          [4, 'C', 4, 'O'],
          [6, 'A', 8, 'O'],
          [8, 'C', 7, 'O'],
          [9, 'B', 5, 'O'],
          [10, 'D', 3, 'O']]
fixed_plan_22 = \
         [[1, 'A', 10, 'O'],
          [2, 'A', 9, 'O'],
          [3, 'C', 10, 'O'],
          [4, 'B', 5, 'O'],
          [5, 'B', 7, 'O'],
          [6, 'B', 9, 'O'],
          [7, 'C', 2, 'O'],
          [8, 'C', 4, 'O'],
          [9, 'A', 6, 'O'],
          [10, 'D', 7, 'O']]
fixed_plan_23 = \
         [[3, 'A', 8, 'O'],
          [4, 'C', 8, 'O'],
          [5, 'B', 4, 'O'],
          [6, 'D', 5, 'O'],
          [7, 'C', 5, 'X'],
          [8, 'A', 3, 'X'],
          [9, 'D', 2, 'X']]
fixed_plan_24 = \
         [[2, 'C', 3, 'O'],
          [3, 'B', 1, 'O'],
          [4, 'C', 3, 'X'],
          [5, 'C', 1, 'O'],
          [6, 'D', 2, 'O'],
          [7, 'B', 1, 'O'],
          [8, 'D', 2, 'O'],
          [9, 'C', 7, 'O'],
          [10, 'A', 1, 'X']]
fixed_plan_25 = \
         [[1, 'B', 7, 'X'],
          [3, 'C', 1, 'O'],
          [6, 'D', 3, 'O'],
          [7, 'A', 7, 'O'],
          [8, 'D', 3, 'X'],
          [9, 'C', 7, 'O'],
          [10, 'C', 9, 'X']]
fixed_plan_26 = \
         [[1, 'A', 6, 'O'],
          [2, 'A', 2, 'O'],
          [3, 'A', 9, 'X'],
          [4, 'D', 1, 'X'],
          [5, 'C', 7, 'O'],
          [6, 'D', 6, 'O'],
          [7, 'B', 5, 'O'],
          [8, 'A', 1, 'O'],
          [9, 'D', 10, 'X'],
          [10, 'A', 6, 'O']]
 
#
#--------------------------------------------------------------------#



#-----Function for Assessing Your Solution---------------------------#
#
# The function in this section will be used to mark your solution.
# Do not change any of the code in this section.
#
# The following function creates a random data set specifying a city
# to be built.  Your program must work for any data set returned by
# this function.  The results returned by calling this function will
# be used as the argument to your build_city function during marking.
# For convenience during code development and marking this function
# also prints the plan for the city to be built to the shell window.
#

city_plan = []
def random_plan(print_plan = True):
    building_probability = 70 # percent
    option_probability = 20 # percent
    from random import randint, choice
    # Create a random list of building instructions
    for site in range(1, len(sites) + 1): # consider each building site
        if randint(1, 100) <= building_probability: # decide whether to build here
            style = choice(['A', 'B', 'C', 'D']) # choose building style
            num_floors = randint(1, 10) # choose number of floors
            if randint(1, 100) <= option_probability: # decide on option's value
                option = 'X'
            else:
                option = 'O'
            city_plan.append([site, style, num_floors, option])
    # Optionally print the result to the shell window
    if print_plan:
        print('\nBuildings to be constructed\n' +
              '(site, style, no. floors, option):\n\n',
              str(city_plan).replace('],', '],\n '))
    # Return the result to the student's build_city function
    return city_plan



#--------------------------------------------------------------------#
#-----Student's Solution---------------------------------------------#
#-----Logic to find out what site, style, height of the buildings----------------------#
import turtle

#making list into a var seperate from the function
plan = city_plan #the list that the logic is going off

#defining how many buildings are in current plan
citybuildings  = len(plan)

#math for building sites and defining them
citybuildings -= 3

#defining variable
b = 0

#making b height
height = b

#Changing site list to variables for ease of acess
Site_1_X = sites[0][1][0]
Site_1_Y = sites[0][1][1]
Site_2_X = sites[1][1][0]
Site_2_Y = sites[1][1][1]
Site_3_X = sites[2][1][0]
Site_3_Y = sites[2][1][1]
Site_4_X = sites[3][1][0]
Site_4_Y = sites[3][1][1]
Site_5_X = sites[4][1][0]
Site_5_Y = sites[4][1][1]
Site_6_X = sites[5][1][0]
Site_6_Y = sites[5][1][1]
Site_7_X = sites[6][1][0]
Site_7_Y = sites[6][1][1]
Site_8_X = sites[7][1][0]
Site_8_Y = sites[7][1][1]
Site_9_X = sites[8][1][0]
Site_9_Y = sites[8][1][1]
Site_10_X = sites[9][1][0]
Site_10_Y = sites[9][1][1]

#chaning height var to change the building height in levels
def height_1():
    global b
    b = 0

def height_2():
    global b
    b = 1

def height_3():
    global b
    b = 2

def height_4():
    global b
    b = 3

def height_5():
    global b
    b = 4
    
def height_6():
    global b
    b = 5
    
def height_7():
    global b
    b = 6

def height_8():
    global b
    b = 7
    
def height_9():
    global b
    b = 8

def height_10():
    global b
    b = 9

def fix_B_Count():
    global b
    b+=1
    
#Calculating how many buildings are their in this list
x = -1 # the reason it is -1 is because this cuts out the base

#defining x and y var for turtle to then go to the location of sites
def buildingsite():       
    buildsite = plan[x][0] 
    if buildsite == (1):
        Site_1_X
        Site_1_Y
        return Site_1_X, Site_1_Y
    elif buildsite == (2) :
        Site_2_X
        Site_2_Y
        return Site_2_X, Site_2_Y
    elif buildsite == (3) :
        Site_3_X
        Site_3_Y
        return Site_3_X, Site_3_Y
    elif buildsite == (4) :
        Site_4_X
        Site_4_Y
        return Site_4_X, Site_4_Y
    elif buildsite == (5) :
        Site_5_X
        Site_5_Y
        return Site_5_X, Site_5_Y
    elif buildsite == (6) :
        Site_6_X
        Site_6_Y
        return Site_6_X, Site_6_Y
    elif buildsite == (7) :
        Site_7_X
        Site_7_Y
        return Site_7_X, Site_7_Y
    elif buildsite == (8) :
        Site_8_X
        Site_8_Y
        return Site_8_X, Site_8_Y
    elif buildsite == (9) :
        Site_9_X
        Site_9_Y
        return Site_9_X, Site_9_Y
    elif buildsite == (10) :
        Site_10_X
        Site_10_Y
        return Site_10_X, Site_10_Y

#defining what style to build from the random plan
def buildingstyle():    
    buildingstyle = plan[x][1] 
    if buildingstyle == 'A' :
        Build_A()
    elif buildingstyle == 'B' :
        Build_B()
    elif buildingstyle == 'C' :
        Build_C()
    elif buildingstyle == 'D' :
        Build_D()

#defining how many levels the building will have
def buildingheight():
    buildingheight = plan[x][2]
    if buildingheight == (1) :
        height_1()
    elif buildingheight == (2) :
        height_2()
    elif buildingheight == (3) :
        height_3()
    elif buildingheight == (4) :
        height_4()
    elif buildingheight == (5) :
        height_5()
    elif buildingheight == (6) :
        height_6()
    elif buildingheight == (7) :
        height_7()
    elif buildingheight == (8) :
        height_8()
    elif buildingheight == (9) :
        height_9()
    elif buildingheight == (10) :
        height_10()

#defining what to build if their is an X or O in the list of data.
#defines if it is in construction for A
def buildingconstruction_A():
    buildingconstruction = plan[x][3]
    if buildingconstruction == ('X'):
        Crane()
    elif buildingconstruction == ('O'):
        A_Roof()
        
#defines if it is in construction for B
def buildingconstruction_B():
    buildingconstruction = plan[x][3]
    if buildingconstruction == ('X'):
        turtle.penup()
        turtle.forward(170)
        turtle.left(90)
        turtle.forward(20)
        turtle.right(90)
        turtle.pendown()
        Crane()
    elif buildingconstruction == ('O'):
        B_Roof()
        
#defines if it is in construction for C
def buildingconstruction_C():
    buildingconstruction = plan[x][3]
    if buildingconstruction == ('X'):
        turtle.penup()
        turtle.forward(3)
        turtle.left(90)
        turtle.forward(100)
        turtle.right(90)
        turtle.left(90)
        turtle.pendown()
        Crane()
    elif buildingconstruction == ('O'):
        C_Roof()
        
#defines if it is in construction for D
def buildingconstruction_D():
    buildingconstruction = plan[x][3]
    if buildingconstruction == ('X'):
        turtle.penup()
        turtle.left(90)
        turtle.forward(90)
        turtle.right(90)
        turtle.left(90)
        turtle.pendown()
        Crane()
    elif buildingconstruction == ('O'):
        D_Roof()

#for times in range(citybuildings): reapeat n amount of times

#--------------------------------------------------------------------#

#setting the number of buildings back to the correct values
def number_of_buildings():
    global citybuildings
    citybuildings  = len(plan) - 2


#changing the x value by 1 to go down the list and change the building style, height and site location
def count():
    global x
    while True:
        x += 1
        buildingstyle()
        if x > citybuildings:
            break

#defining what to build on every level of building style A
def building_height_count_A():
    buildingheight()
    if b == (0):
        turtle.penup()
        turtle.right(90)
        turtle.forward(50)
        turtle.left(90)
        turtle.left(180)
        turtle.forward(130)
        turtle.pendown()
    elif b == (1):
        build_2_floor()
        turtle.penup()
        turtle.right(180)
        turtle.forward(20)
        turtle.left(180)
        turtle.left(90)
        turtle.forward(100)
        turtle.pendown()
    elif b == (2):
        build_2_floor()
        A_Stories_2()
    elif b == (3):
        build_2_floor()
        A_Stories_2()
        for i in range(1):
            A_Stories_2_1()
    elif b == (4):
        build_2_floor()
        A_Stories_2()
        for i in range(2):
            A_Stories_2_1()
    elif b == (5):
        build_2_floor()
        A_Stories_2()
        for i in range(3):
            A_Stories_2_1()
    elif b == (6):
        build_2_floor()
        A_Stories_2()
        for i in range(4):
            A_Stories_2_1()
    elif b == (7):
        build_2_floor()
        A_Stories_2()
        for i in range(5):
            A_Stories_2_1()
    elif b == (8):
        build_2_floor()
        A_Stories_2()
        for i in range(6):
            A_Stories_2_1()
    elif b == (9):
        build_2_floor()
        A_Stories_2()
        for i in range(7):
            A_Stories_2_1()
    elif b == (10):
        build_2_floor()
        A_Stories_2()
        for i in range(8):
            A_Stories_2_1()


    
#defining what to build on every level of building style B
def building_height_count_B():
    buildingheight()
    fix_B_Count()
    if b == (1):
        fourth_floor_fix_level_one()  
    elif b == (2):
        first_floor_B()
        fourth_floor_fix_level_two()
    elif b == (3):
        first_floor_B()
        second_floor_c()
        fourth_floor_fix_level_three()
    elif b == (4): 
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        fourth_floor_run()
    elif b == (5):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(2):
            fourth_floor_run()
    elif b == (6):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(3):
            fourth_floor_run()
    elif b == (7):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(4):
            fourth_floor_run()
    elif b == (8):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(5):
            fourth_floor_run()
    elif b == (9):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(6):
            fourth_floor_run()
    elif b == (10):
        first_floor_B()
        second_floor_c()
        move_fourth_floor_up()
        for i in range(7):
            fourth_floor_run()

#defining what to build on every level of building style C           
def building_height_count_C():
    buildingheight()
    if b == (0):
        turtle.pendown()
        turtle.forward(5)
            #if b < 11:
    elif b == (1):
        for i in range(b):
           move_c_stories()
    elif b == (2):
        for i in range(b):        
            move_c_stories()
    elif b == (3):
        for i in range(b):
            move_c_stories()
    elif b == (4):
        for i in range(b):
            move_c_stories()
    elif b == (5):
        for i in range(b):
            move_c_stories()
    elif b == (6):
        for i in range(b):
            move_c_stories()
    elif b == (7):
        for i in range(b):
            move_c_stories()
    elif b == (8):
        for i in range(b):
            move_c_stories()
    elif b == (9):
        for i in range(b):
            move_c_stories()
    elif b == (10):
        for i in range(b):
            move_c_stories()

#defining what to build on every level of building style D
def building_height_count_D():
    buildingheight()
    for i in range(b):
            #if b < 11:
        if b == (1):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories()        
        elif b == (2):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (3):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (4):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (5):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (6):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (7):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (8):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (9):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories() 
        elif b == (10):
            turtle.color('black', 'green')
            turtle.penup()
            turtle.right(180)
            turtle.forward(30)
            turtle.right(90)
            D_Stories()

#----------------------Turtle Building----------------------#
from turtle import *

#Style A building run command
#LAPD Building | Blade Runner 2048
def Build_A():
    A_Base()
    building_height_count_A()
    buildingconstruction_A()

    
#Style B building run command
#Empire state building | Men In Black III
def Build_B():
    B_Base()
    building_height_count_B()
    buildingconstruction_B()

#Style C building run command
#Big Ben | Independence day
def Build_C():
    C_Base()
    building_height_count_C()
    buildingconstruction_C()

#Style D building run command
#The Avengers HQ | The Avengers
def Build_D():
    D_Base()        
    building_height_count_D()
    buildingconstruction_D()

#building Style A 1st floor or base 
def A_Base():
    turtle.penup()
    turtle.goto(buildingsite())
    turtle.pendown()
    turtle.color('black', 'grey')
    turtle.begin_fill()
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.penup()
    turtle.forward(20)
    turtle.pendown()
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.left(90)
    turtle.forward(35)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(35)
    turtle.end_fill()

#builds second floor for A
def build_2_floor():
    turtle.color('black', 'grey')
    turtle.penup()
    turtle.left(180)
    turtle.forward(50)
    turtle.right(180)
    turtle.right(90)
    turtle.forward(130)
    turtle.pendown()
    A_Stories_1()   
    
#makes 2 floor
def A_Stories_1():
    turtle.color('black', 'grey')
    turtle.begin_fill()
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.penup()
    turtle.forward(20)
    turtle.pendown()
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.left(90)
    turtle.forward(35)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(35)
    turtle.end_fill()

#Builds 3rd floor
def A_Stories_2_1():
    turtle.penup()
    turtle.left(180)
    turtle.forward(105)
    turtle.right(180)
    turtle.right(90)
    turtle.forward(20)
    turtle.left(90)
    turtle.right(90)
    turtle.pendown()
    A_Stories_2()
            
#makes 3rd floor
def A_Stories_2():
    turtle.color('black', 'grey')
    turtle.left(180)
    turtle.penup()
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(30)
    turtle.right(90)
    turtle.pendown()
    turtle.begin_fill()
    turtle.left(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(30)
    turtle.right(90)
    turtle.forward(140)
    turtle.right(90)
    turtle.forward(30)
    turtle.right(90)
    turtle.forward(70)
    turtle.end_fill()
    reapeat_window_A_1()
    windows_A_2()
    reapeat_window_A_2()
    turtle.right(180)
    
#makes reapears window_A_2
def reapeat_window_A_2():
    turtle.penup()
    turtle.right(180)
    turtle.forward(105)
    turtle.right(180)
    turtle.left(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.pendown()
    for i in range(3):
        windwos_A_1()
        turtle.penup()
        turtle.right(180)
        turtle.forward(80)
        turtle.left(180)
        turtle.left(90)
        turtle.forward(10)
        turtle.right(90)
        turtle.pendown()

#repeats window A 1
def reapeat_window_A_1():
    for i in range(3):
        windwos_A_1()
        turtle.penup()
        turtle.right(180)
        turtle.forward(80)
        turtle.right(180)
        turtle.left(90)
        turtle.forward(10)
        turtle.right(90)
        turtle.pendown()

#draws window A2
def windows_A_2():
    turtle.color('black', 'black')
    turtle.penup()
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(10)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(30)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(30)
    turtle.end_fill()

#draws windows A1
def windwos_A_1():
    turtle.color('black', 'black')
    turtle.penup()
    turtle.forward(65)
    turtle.right(90)
    turtle.forward(10)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.end_fill()
    
#building the roof of style A
def A_Roof():
    turtle.color('black', 'grey')
    turtle.penup()
    turtle.left(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(140)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(90)
    turtle.right(60)
    turtle.forward(40)
    turtle.right(30)
    turtle.right(90)
    turtle.forward(240)
    turtle.right(135)
    turtle.forward(49)
    turtle.left(45)
    turtle.right(90)
    turtle.forward(150)
    turtle.end_fill()
    A_Roof_LAPD()


#draws LAPD on top of the building
def A_Roof_LAPD():
    turtle.color('black', 'grey')
    turtle.right(180)
    #draws the L
    turtle.penup()
    turtle.forward(15)
    turtle.pendown()
    turtle.pensize(5)
    turtle.left(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.forward(25)
    turtle.right(180)
    turtle.forward(25)
    turtle.left(90)
    turtle.forward(15)
    #draws the A
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()
    turtle.left(60)
    turtle.forward(25)
    turtle.right(120)
    turtle.forward(25)
    turtle.right(180)
    turtle.forward(10)
    turtle.left(60)
    turtle.forward(15)
    #draws the P
    turtle.left(90)
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()
    turtle.right(90)
    turtle.right(180)
    turtle.penup()
    turtle.forward(30)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(5)
    turtle.pendown()
    turtle.forward(20)
    turtle.right(90)
    turtle.right(90)
    turtle.forward(13)
    turtle.left(90)
    turtle.pensize(5)
    turtle.circle(8,180)
    #draws the D
    turtle.left(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.penup()
    turtle.forward(20)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(30)
    turtle.right(180)
    turtle.forward(30)
    turtle.right(180)
    turtle.right(90)
    turtle.circle(15,180)
    turtle.pensize(1)
    turtle.left(180)
    
#building styles B Base
def B_Base():
    turtle.penup()
    turtle.goto(buildingsite())
    turtle.pendown()
    turtle.color('black', 'grey')
    turtle.begin_fill()
    turtle.forward(120)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(240)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(120)
    turtle.end_fill()
    turtle.penup()
    turtle.left(180)
    turtle.forward(136)
    turtle.right(180)
    turtle.forward(10)
    turtle.pensize(1)
    B_Base_Windows_Space()
    turtle.goto(buildingsite())
    turtle.penup()
    turtle.left(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.pendown()

#Makes the 2nd floor for B
def first_floor_B():
    turtle.color('black', 'grey')
    turtle.begin_fill()
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(200)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.penup()
    turtle.left(180)
    turtle.forward(98)
    turtle.pendown()
    turtle.right(90)
    b_windows_2_run()

#Makes 3 floor for B
def second_floor_c():
    turtle.color('black', 'grey')
    turtle.penup()
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(115)
    turtle.right(180)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(72)
    turtle.left(90)
    turtle.forward(170)
    turtle.left(90)
    turtle.forward(72)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.penup()
    turtle.left(180)
    turtle.forward(98)
    turtle.pendown()
    turtle.right(90)
    turtle.penup()
    turtle.right(180)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(2)
    turtle.right(90)
    turtle.pendown()
    turtle.pensize(1)
    B_Window_3()

#repositions level three to be correct
def fourth_floor_fix_level_three():
    turtle.left(90)
    turtle.forward(42)
    turtle.right(90)
    turtle.right(180)
    turtle.forward(115)
    turtle.left(180)

#repositions level two to be correct
def fourth_floor_fix_level_two():
    turtle.left(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.right(180)
    turtle.forward(135)
    turtle.left(180)
    
#repositions level one to be correct
def fourth_floor_fix_level_one():
    turtle.penup()
    turtle.right(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(20)
    turtle.left(180)
    turtle.forward(55)
    turtle.right(180)
    
#repositions fourth floor
def move_fourth_floor_up():
    turtle.penup()
    turtle.left(90)
    turtle.forward(42)
    turtle.left(90)
    turtle.forward(120)
    turtle.right(180)
    turtle.pendown()

#Builds fourth floor
def fourth_floor_run():
    fourth_floor()
    lift_up_fourth_floor_window()
    fourth_floor_window_reapeat()
    fourth_floor_window_onces()
    fourth_floor_window_last_one()
    turtle.left(90)
    turtle.forward(90)
    turtle.left(90)
    turtle.forward(19)
    turtle.left(90)

# makes windows reapeat for fourth floor
def fourth_floor_window_reapeat():
    for i in range(1):
        turtle.pensize(1)
        fourth_floor_window()
        turtle.penup()
        turtle.forward(47)
        turtle.left(90)

#reapeats the fourth floor windows four times
def fourth_floor_window_onces():
    for i in range(4):
        turtle.pensize(1)
        fourth_floor_window()
        turtle.penup()
        turtle.forward(37)
        turtle.left(90)

#reapeats the fourth floor windows four times. also repositions
def fourth_floor_window_last_one():
    for i in range(1):
        turtle.penup()
        turtle.right(90)
        turtle.forward(9)
        turtle.left(90)
        turtle.pendown()
        fourth_floor_window()
        turtle.left(90)

#makes fourth floor
def fourth_floor():
    turtle.color('black', 'grey')
    turtle.penup()
    turtle.left(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(130)
    turtle.left(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.penup()
    turtle.left(180)
    turtle.forward(98)
    turtle.pendown()
    turtle.right(90)
    turtle.penup()
    turtle.right(180)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(28)

#repositions fourth floor windows
def lift_up_fourth_floor_window():
    turtle.left(90)
    turtle.forward(19)

#Makes fourth  floor windows
def fourth_floor_window():
    turtle.pensize(1)
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.left(90)
    turtle.penup()
    turtle.forward(25.5)
    turtle.right(90)
    turtle.forward(29)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(29)
    turtle.pendown()
    turtle.end_fill()
    turtle.left(180)
    turtle.color('grey', 'black')    
    turtle.forward(7.5)
    turtle.left(90)
    turtle.forward(16)
    turtle.right(90)
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(16)
    turtle.left(90)
    turtle.forward(7.5)
    turtle.left(90)
    turtle.forward(16)
    turtle.right(90)
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(30)
    turtle.left(90) 

#makes third floor
def third_floor():
    turtle.color('black', 'grey')
    turtle.penup()
    turtle.forward(17)
    turtle.left(90)
    turtle.forward(105)
    turtle.left(90)
    turtle.forward(135)
    turtle.right(180)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(130)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.penup()
    turtle.left(180)
    turtle.forward(98)
    turtle.pendown()
    turtle.right(90)
    turtle.penup()
    turtle.right(180)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(2)
    turtle.right(90)
    turtle.pendown()
    turtle.left(90)

#builds b windows 3
def B_Window_3():
    turtle.pensize(1)
    B_Base_Windows_Reapeat_3_1()
    B_Base_Windows_Space_3()
    turtle.left(90)
    turtle.forward(12)
    turtle.left(90)
    turtle.right(90)
    turtle.right(90)
    B_Base_Windows_Reapeat_3_1()
    turtle.left(90)

#makes a space inbeerween windows for left of building
def B_Base_Windows_Space_3():
    for i in range(1):
        turtle.pensize(1)
        turtle.left(90)
        turtle.forward(11)
        turtle.right(90)
        B_Base_Windows_Reapeat_3_2()

#repeat windows for middle of building
def B_Base_Windows_Reapeat_3_2():
    for i in range(4):
        turtle.pensize(1)
        windows_b_three()
        turtle.forward(10)
        turtle.left(90)
        turtle.forward(9)
        turtle.right(90)
        turtle.right(90)

#repeats windows for right of building
def B_Base_Windows_Reapeat_3_1():
    for i in range(2):
        turtle.pensize(1)
        windows_b_three()
        turtle.forward(10)
        turtle.left(90)
        turtle.forward(9)
        turtle.right(90)
        turtle.right(90)

#builds windows B three
def windows_b_three():
    turtle.left(180)
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.forward(91)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(80)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.end_fill()
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(180)
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(180)
    turtle.forward(100)
    turtle.left(90)
    
#Making window style 3 run
def b_windows_2_run():
    turtle.pensize(1)
    B_Base_Windows_Reapeat_2()
    turtle.right(90)
    turtle.forward(12)
    turtle.left(90)
    B_Base_Windows_Reapeat_2_1()
    turtle.right(90)
    turtle.forward(12)
    turtle.left(90)
    B_Base_Windows_Reapeat_2()
    turtle.right(90)

#makes windows repeat left
def B_Base_Windows_Space_2():
    for i in range(3):
        turtle.pensize(1)
        B_Base_Windows_Reapeat_2()
        turtle.right(90)
        turtle.forward(10)
        turtle.left(90)

#makes windows reapeat middle
def B_Base_Windows_Reapeat_2():
    for i in range(3):
        turtle.pensize(1)
        windows_b_two()
        turtle.forward(9)

#makes windows repeat right
def B_Base_Windows_Reapeat_2_1():
    for i in range(4):
        turtle.pensize(1)
        windows_b_two()
        turtle.forward(9)
        
#window b design  
def windows_b_two():
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.forward(90)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(99)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.end_fill()
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(180)
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(180)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)

#making space for the windows to build 10 pixels over 3 times 
def B_Base_Windows_Space():
    for i in range(3):
        turtle.forward(10)
        B_Base_Windows_Reapeat()

#reapeating the window building 4 times 
def B_Base_Windows_Reapeat():
    for i in range(4):
        B_Base_Windows()
        
#Making window style 1    
def B_Base_Windows():
    turtle.penup()
    turtle.left(90)
    turtle.pendown()
    turtle.color('black', 'black')
    turtle.begin_fill()
    turtle.forward(40)
    turtle.right(90)
    turtle.penup()
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(40)
    turtle.end_fill()
    turtle.color('grey', 'green')
    turtle.right(90)
    turtle.penup()
    turtle.forward(7.5)
    turtle.right(90)
    turtle.forward(1)
    turtle.pendown()
    turtle.forward(40)
    turtle.left(90)
    turtle.forward(7.5)
    turtle.right(180)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(16)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(16)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(16)
    turtle.left(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(16)
    turtle.penup()
    turtle.forward(3)
    turtle.right(90)
    turtle.forward(1)
    turtle.left(90)

#Makes b stories
def B_Stories():
    turtle.penup()
    turtle.left(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.pendown()
    turtle.color('black', 'blue')
    turtle.begin_fill()
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(200)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.penup()

#Makes B roof
def B_Roof():
    turtle.penup()
    turtle.left(90)
    turtle.forward(50)
    turtle.pendown()
    turtle.left(90)
    turtle.color('black', 'grey')
    turtle.begin_fill()
    turtle.forward(30)
    for i in range(5):
        turtle.right(90)
        turtle.forward(10)
        turtle.right(90)
        turtle.forward(10)
        turtle.right(180)
    turtle.right(90)
    turtle.forward(40)
    turtle.right(45)
    turtle.forward(20)
    turtle.left(45)
    turtle.forward(40)
    turtle.right(90)
    turtle.forward(2)
    turtle.right(90)
    turtle.forward(40)
    turtle.left(45)
    turtle.forward(20)
    turtle.right(45)
    turtle.forward(40)
    for i in range(4):
        turtle.left(90)
        turtle.forward(10)
        turtle.right(90)
        turtle.forward(10)
    turtle.left(90)
    turtle.forward(9.5)
    turtle.right(90)
    turtle.forward(10)
    turtle.end_fill()
    turtle.left(90)
    turtle.pensize(1)
    roof_window()
    turtle.pensize(1)

#makes roof window for B
def roof_window():
    turtle.color('grey', 'black')
    turtle.begin_fill()
    turtle.pendown()
    turtle.left(180)
    turtle.forward(75)
    turtle.right(90)
    turtle.forward(90)
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(90)
    turtle.end_fill()
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(110)
    turtle.right(180)
    turtle.forward(110)
    turtle.color('black', 'black')
    turtle.left(90)
    turtle.forward(65)
    turtle.right(180)
    turtle.forward(100)
    turtle.right(180)
    turtle.pensize(1)

#builds C base
def C_Base():
    #builds oustide of boc
    turtle.pensize(2)
    turtle.penup()
    turtle.goto(buildingsite())
    turtle.pendown()
    turtle.color('black', 'Goldenrod')
    turtle.begin_fill()
    turtle.forward(60)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.pensize(2)
    turtle.forward(120)
    turtle.left(90)
    turtle.forward(50)
    turtle.end_fill()
    turtle.left(90)
    turtle.forward(10)
    #starts building lines and details on the base
    turtle.left(90)
    turtle.pensize(2)
    turtle.forward(50)
    turtle.pensize(1)
    turtle.right(90)
    turtle.forward(100)
    turtle.right(90)
    turtle.pensize(2)
    turtle.forward(50)
    turtle.pensize(1)
    turtle.right(180)
    turtle.forward(5)
    turtle.right(180)
    turtle.right(90)
    turtle.pensize(2)
    turtle.forward(110)
    turtle.right(180)
    turtle.pensize(2)
    turtle.forward(120)
    turtle.pensize(1)
    turtle.left(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.pensize(2)
    turtle.forward(120)
    turtle.pensize(1)
    turtle.right(90)
    turtle.forward(5)
    turtle.right(90)
    turtle.pensize(2)
    turtle.forward(120)
    turtle.right(180)
    turtle.pensize(1)
    turtle.forward(90)
    turtle.right(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.pensize(4)
    turtle.forward(15)
    turtle.penup()
    turtle.right(90)
    turtle.forward(10)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(15)
    turtle.pensize(1)
    turtle.left(90)
    turtle.penup()
    turtle.forward(40)
    turtle.pendown()
    turtle.left(90)
    turtle.pensize(4)
    turtle.forward(15)
    turtle.pensize(1)
    turtle.right(90)
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()
    turtle.right(90)
    turtle.pensize(4)
    turtle.forward(15)
    turtle.pensize(2)
    turtle.penup()
    turtle.forward(30)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(90)
    turtle.right(180)
    turtle.forward(90)
    turtle.left(90)
    turtle.penup()
    turtle.forward(30)
    turtle.right(180)
    turtle.pensize(1)
    turtle.pendown()

#repositions level 1 c
def fix_level_1_c():
    turtle.left(90)
    turtle.penup()
    turtle.forward(1000)
    turtle.pendown()
    turtle.right(90)

#repositions c stories
def move_c_stories():
    C_Stories()
    turtle.left(90)
    turtle.penup()
    turtle.forward(60)
    turtle.pendown()
    turtle.right(90)

#makes C Stories
def C_Stories():
    turtle.color('black', 'Goldenrod')
    turtle.begin_fill()
    turtle.pensize(2)
    turtle.right(180)
    turtle.penup()
    turtle.forward(20)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(45)
    turtle.left(90)
    turtle.forward(120)
    turtle.left(90)
    turtle.forward(45)
    turtle.left(90)
    turtle.forward(90)
    turtle.end_fill()
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(45)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(45)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(120)
    turtle.right(90)
    turtle.forward(10)
    turtle.pensize(4)
    turtle.penup()
    turtle.right(90)
    turtle.forward(30)
    turtle.right(90)
    turtle.forward(18)
    turtle.pendown()
    turtle.forward(15)
    turtle.left(90)
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()    
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.penup()
    turtle.forward(40)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(15)
    turtle.pensize(1)
    
#makes C Roof
def C_Roof():
    turtle.color('black', 'Goldenrod')
    #first small rectangle on roof
    turtle.pensize(2)
    turtle.begin_fill()
    turtle.penup()
    turtle.right(180)
    turtle.forward(25)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(22)
    turtle.left(90)
    turtle.forward(120)
    turtle.left(90)
    turtle.forward(22)
    turtle.left(90)
    turtle.forward(90)
    turtle.end_fill()
    #windows for the smalle rectangele
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(22)
    turtle.left(180)
    turtle.forward(22)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    rectangle_window_repeat_C()
    turtle.left(90)
    turtle.penup()
    turtle.forward(5)
    turtle.right(90)
    turtle.pendown()
    turtle.forward(20)
    turtle.right(180)
    turtle.forward(21)
    turtle.color('black', 'Goldenrod')
    turtle.begin_fill()
    turtle.right(180)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(130)
    turtle.right(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(120)
    turtle.end_fill()
    turtle.left(180)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(70)
    turtle.left(90)
    turtle.forward(90)
    turtle.left(90)
    turtle.forward(70)
    turtle.left(90)
    turtle.forward(30)
    clock()
    
#Makes Clock
def clock():
    turtle.penup()
    turtle.forward(15)
    turtle.left(90)
    turtle.forward(5)
    turtle.right(90)
    turtle.pendown()
    turtle.color('black', 'white')
    turtle.begin_fill()
    turtle.circle(30)
    turtle.end_fill()
    turtle.left(90)
    turtle.penup()
    turtle.forward(30)
    turtle.pendown()
    turtle.pensize(5)
    turtle.circle(1)
    turtle.pensize(1)
    turtle.right(90)
    turtle.penup()
    turtle.forward(20)
    turtle.pendown()
    turtle.left(90)
    turtle.pensize(2)
    turtle.penup()
    turtle.circle(20)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(25)
    turtle.left(90)
    turtle.forward(5)
    turtle.pendown()
    #building the tII or 12 on the clock
    turtle.forward(10)
    turtle.right(180)
    turtle.forward(5)
    turtle.left(90)
    turtle.forward(3)
    turtle.right(180)
    turtle.forward(6)
    turtle.right(180)
    turtle.forward(3)
    turtle.right(90)
    turtle.forward(3)
    turtle.right(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.forward(2)
    turtle.left(90)
    turtle.forward(2)
    turtle.right(180)
    turtle.forward(10)
    turtle.left(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.penup()
    #draw the I or 1 on the clock
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(5)
    turtle.pendown()
    turtle.right(40)
    turtle.forward(10)
    turtle.left(40)
    #draw the II or 2 on the clock
    turtle.left(90)
    turtle.penup()
    turtle.forward(15)
    turtle.pendown()
    turtle.right(90)
    turtle.right(60)
    turtle.forward(10)
    turtle.right(30)
    turtle.left(90)
    turtle.penup()
    turtle.forward(3)
    turtle.left(90)
    turtle.forward(2)
    turtle.right(90)
    turtle.pendown()
    turtle.left(115)
    turtle.forward(10)
    #drwar the III on the clock
    turtle.right(25)
    turtle.right(90)
    turtle.penup()
    turtle.forward(10)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(10)
    turtle.left(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(10)
    #Draws the IV on the clock
    turtle.left(90)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.left(60)
    turtle.forward(7)
    turtle.right(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.right(110)
    turtle.forward(7)
    turtle.left(150)
    turtle.forward(7)
    turtle.left(30)
    turtle.penup()
    turtle.forward(5)
    turtle.right(130)
    turtle.forward(5)
    turtle.pendown()
    #Draws the V on the clock
    turtle.right(55)
    turtle.forward(7)
    turtle.left(150)
    turtle.forward(7)
    turtle.left(30)
    turtle.right(130)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    #drwaws the VI on the clock
    turtle.right(90)
    turtle.forward(7.5)
    turtle.left(150)
    turtle.forward(10)
    turtle.left(30)
    turtle.right(90)
    turtle.penup()
    turtle.forward(2)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(8)
    turtle.left(90)
    turtle.penup()
    turtle.forward(6)
    turtle.pendown()
    #Draws the VII on the clock
    turtle.left(70)
    turtle.forward(7)
    turtle.right(100)
    turtle.penup()
    turtle.forward(3)
    turtle.right(45)
    turtle.forward(3)
    turtle.pendown()
    turtle.right(70)
    turtle.forward(7)
    turtle.right(20)
    turtle.left(120)
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.left(70)
    turtle.forward(7)
    turtle.left(20)
    turtle.right(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.right(115)
    turtle.forward(10)
    #draws the IE
    turtle.left(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.left(70)
    turtle.forward(10)
    turtle.right(90)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.right(90)
    turtle.forward(10)
    turtle.right(180)
    turtle.forward(5)
    turtle.left(90)
    turtle.forward(3)
    turtle.left(180)
    turtle.forward(6)
    turtle.penup()
    turtle.forward(5)
    turtle.pendown()
    turtle.left(90)
    turtle.forward(4)
    #draws the t
    turtle.left(180)
    turtle.forward(7)
    turtle.right(180)
    turtle.forward(3.5)
    turtle.left(90)
    turtle.forward(2)
    turtle.right(180)
    turtle.forward(4)
    #draws the Ti
    turtle.penup()
    turtle.forward(3)
    turtle.pendown()
    turtle.right(110)
    turtle.forward(5)
    #builds clock hand
    turtle.penup()
    turtle.left(25)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.pendown()
    turtle.forward(20)
    turtle.right(180)
    turtle.forward(20)
    turtle.right(180)
    turtle.left(90)
    turtle.forward(15)
    #builds the top of roof
    turtle.color('black', 'Goldenrod')
    turtle.penup()
    turtle.forward(34)
    turtle.left(90)
    turtle.forward(32)
    turtle.pendown()
    turtle.begin_fill()
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(100)
    turtle.end_fill()
    turtle.right(180)
    turtle.forward(20)
    turtle.left(180)
    turtle.left(90)
    rectangle_window_repeat_C_2()
    #builds the triangle on top of building
    turtle.color('black', 'Goldenrod')
    turtle.begin_fill()
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(14)
    turtle.right(90)
    turtle.right(40)
    turtle.forward(80)
    turtle.right(100)
    turtle.forward(80)
    turtle.right(130)
    turtle.forward(100)
    turtle.end_fill()
    turtle.right(180)      

#repears windows for C 2
def rectangle_window_repeat_C_2():
    for i in range(4):  
        small_window_C()
        turtle.left(180)
        turtle.penup()
        turtle.forward(15)
        turtle.right(90)
        turtle.pendown()   

#repeats windows C
def rectangle_window_repeat_C():
    for i in range(5):  
        small_window_C()
        turtle.left(180)
        turtle.penup()
        turtle.forward(15)
        turtle.right(90)
        turtle.pendown()

#makes small window c
def small_window_C():
    turtle.color('black', 'black')
    turtle.begin_fill()
    turtle.forward(10)
    turtle.circle(6,180)
    turtle.forward(10)
    turtle.left(90)
    turtle.forward(10)
    turtle.end_fill()

#builds D base
def D_Base():
    turtle.penup()    
    turtle.color('white', 'black')
    turtle.goto(buildingsite())
    turtle.begin_fill()
    turtle.pendown()
    turtle.forward(120)
    turtle.left(100)
    turtle.forward(30)
    turtle.right(10)
    turtle.left(90)
    turtle.forward(235)
    turtle.left(90)
    turtle.forward(29.5)
    turtle.end_fill()
    turtle.left(90)
    turtle.forward(121)
    turtle.right(90)
    turtle.right(180)
    turtle.left(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(25)
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(25)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(25)
    turtle.right(180)
    turtle.forward(25)
    turtle.left(90)
    turtle.forward(10)
    turtle.right(90)


#--------------------------------------------------------------------#
#values to make the tower lean
z=230
o=115
i=0
j=4

#variable to change the distance and hight for the lean
def fix_d():
    global i
    global j
    i+=1
    if i == (1):
        lean_1()
    elif i == (2):
        lean_2()
    elif i == (3):
        lean_3()
    elif i == (4):
        lean_4()
    elif i == (5):
        lean_5()
    elif i == (6):
        lean_6()
    elif i == (7):
        lean_7()
    elif i == (8):
        lean_8()
    
#this is defining the variables to be correct for the lean
def lean_1():
    global z
    global o
    z=231
    o=115
    z-=5
    o-=5

def lean_2():
    global z
    global o
    z=225
    o=110
    z-=5
    o-=5

def lean_3():
    global z
    global o
    z=213
    o=100
    
def lean_4():
    global z
    global o
    z=209
    o=95
    z-=5
    o-=5

def lean_5():
    global z
    global o
    z=200
    o=90
    z-=5
    o-=5

def lean_6():
    global z
    global o
    z=195
    o=80
    z-=5
    o-=5
    
def lean_7():
    global z
    global o
    z=195
    o=75
    z-=5
    o-=5

def lean_8():
    global z
    global o
    z=190
    o=75
    z-=5
    o-=5

def reset_lean():
    global z
    global o
    global i
    z=230
    o=115
    i=0

#--------------------------------------------------------------------#
#makes D Stories
def D_Stories():
    turtle.color('white', 'black')
    turtle.begin_fill()
    turtle.pendown()
    turtle.forward(o)
    turtle.left(100)
    turtle.forward(30)
    turtle.right(10)
    turtle.left(90)
    turtle.forward(z)
    turtle.left(90)
    turtle.forward(29.5)
    turtle.end_fill()
    turtle.left(90)
    turtle.forward(120)
    turtle.right(90)
    turtle.right(90)
    turtle.forward(100)
    #builds windows
    turtle.right(90)
    turtle.forward(30)
    for i in range(j):
        turtle.right(90)
        turtle.forward(20)
        turtle.right(90)
        turtle.forward(30)
        turtle.left(90)
        turtle.forward(20)
        turtle.left(90)
        turtle.forward(30)
    turtle.penup()
    turtle.left(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.right(180)
    turtle.forward(30)
    turtle.left(180)    
    turtle.right(180)
    turtle.pendown()
    fix_d()

def D_Roof():
    turtle.color('white', 'grey')
    turtle.penup()
    turtle.right(180)
    turtle.forward(28)
    turtle.left(90)
    turtle.forward(100)
    turtle.right(90)
    turtle.begin_fill()
    turtle.pendown()
    turtle.forward(100)
    turtle.right(90)
    turtle.forward(80)
    turtle.right(70)
    turtle.forward(30)
    turtle.left(70)
    turtle.forward(110)
    turtle.right(90)
    turtle.penup()
    turtle.forward(50)
    turtle.pendown()
    turtle.left(90)
    turtle.circle(25,180)
    turtle.right(90)
    turtle.right(180)
    turtle.penup()
    turtle.forward(50)
    turtle.pendown()
    turtle.left(90)
    turtle.right(180)
    turtle.forward(70)
    turtle.left(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(131)
    turtle.end_fill()
    Draw_A_on_Tower()
    reset_lean()


def Draw_A_on_Tower():
    turtle.penup()
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(40)
    turtle.left(70)
    turtle.pendown()
    turtle.pensize(4)
    turtle.forward(50)
    turtle.right(140)
    turtle.forward(50)
    turtle.right(180)
    turtle.forward(20)
    turtle.left(70)
    turtle.forward(20)
    turtle.right(180)
    turtle.penup()
    turtle.forward(45)
    turtle.pendown()
    turtle.left(90)
    turtle.pensize(2)  
    turtle.circle(35)
    turtle.pensize(1)
    turtle.right(90)

def Crane():
    #moves crane to top of the roof
    turtle.penup()
    turtle.left(90)
    turtle.forward(30)
    turtle.left(90)
    turtle.forward(180)
    turtle.pendown()
    #builds base of crane
    turtle.color('black', 'orange')
    turtle.begin_fill()    
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(15)
    turtle.right(90)
    turtle.forward(20)
    turtle.end_fill()
    #builds the cabin for the crane
    turtle.right(90)
    turtle.forward(15)
    turtle.left(90)
    turtle.color('black', 'orange')
    turtle.begin_fill()    
    turtle.forward(30)
    turtle.right(90)
    turtle.forward(40)
    turtle.right(90)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(40)
    turtle.right(90)
    turtle.forward(40)
    turtle.end_fill()
    #draws the window
    turtle.penup()
    turtle.right(90)
    turtle.forward(15)
    turtle.pendown()
    turtle.color('white', 'black')
    turtle.begin_fill()  
    turtle.right(90)
    turtle.forward(35)
    turtle.left(120)
    turtle.forward(25)
    turtle.left(60)
    turtle.forward(22)
    turtle.left(90)
    turtle.forward(22)
    turtle.end_fill()
    turtle.right(180)
    turtle.forward(11)
    turtle.right(90)
    turtle.forward(20)
    turtle.left(90)
    turtle.forward(12)
    turtle.right(180)
    turtle.forward(22)
    turtle.right(180)
    turtle.penup()
    turtle.forward(25)
    turtle.pendown()
    #builds the crane part
    turtle.pensize(2)
    turtle.color('black', 'black')
    turtle.right(45)
    turtle.forward(70)
    turtle.right(90)
    turtle.forward(20)
    turtle.right(90)
    turtle.forward(65)
    turtle.right(180)
    turtle.forward(10)
    turtle.left(180)
    turtle.right(120)
    turtle.forward(25)
    turtle.right(120)
    turtle.forward(25)
    turtle.left(120)
    turtle.forward(25)
    turtle.right(110)
    turtle.penup()
    turtle.forward(45)
    turtle.left(30)
    turtle.left(90)
    turtle.forward(20)
    turtle.pendown()
    #draws the circle on top of crane
    turtle.circle(10)
    turtle.right(205)
    #draws Rope
    turtle.pensize(1)
    turtle.forward(70)
    turtle.left(90)
    #Draws the crane pickup point
    turtle.color('black', 'black')
    turtle.begin_fill()
    turtle.pensize(2)
    turtle.forward(10)
    turtle.right(90)
    turtle.forward(10)
    turtle.right(20)
    turtle.forward(10)
    turtle.right(70)
    turtle.forward(10)
    turtle.right(70)
    turtle.forward(15)
    turtle.right(20)
    turtle.forward(5)
    turtle.right(90)
    turtle.forward(10)
    turtle.end_fill()
    turtle.penup()
    turtle.right(90)
    turtle.forward(24)
    turtle.pendown()
    turtle.left(90)
    turtle.circle(2)
    turtle.penup()
    turtle.forward(2)
    turtle.pendown()
    turtle.left(90)
    turtle.penup()
    turtle.right(180)
    turtle.forward(1)
    turtle.left(180)
    turtle.pendown()
    turtle.right(180)
    turtle.forward(5)
    turtle.right(90)
    turtle.penup()
    turtle.forward(6)
    turtle.pendown()
    turtle.left(90)
    turtle.circle(3,180)
    turtle.right(90)



        
#
#  Complete the assignment by replacing the dummy function below with
#  your own "build_city" function.

# Erect buildings as per the provided city plan
def build_city(plan):
    global city_plan
    city_plan = plan
    number_of_buildings()
    count()
    buildingsite()
    pass
#--------------------------------------------------------------------#



#-----Main Program---------------------------------------------------#
#
# This main program sets up the background, ready for you to start
# building your city.  Do not change any of this code except
# as indicated by the comments marked '*****'.
#

# Set up the drawing canvas
# ***** Change the default argument to False if you don't want to
# ***** display the coordinates and building sites
create_drawing_canvas()

# Control the drawing speed
# ***** Modify the following argument if you want to adjust
# ***** the drawing speed
speed('fastest')

# Decide whether or not to show the drawing being done step-by-step
# ***** Set the following argument to False if you don't want to wait
# ***** while the cursor moves around the screen
tracer(False)

# Give the drawing canvas a title
# ***** Replace this title with your name and/or a description of
# ***** your city
title("SCIFI City")

### Call the student's function to build the city
### ***** While developing your program you can call the build_city
### ***** function with one of the "fixed" data sets, but your
### ***** final solution must work with "random_plan()" as the
### ***** argument to the build_city function.  Your program must
### ***** work for any data set that can be returned by the
### ***** random_plan function.
#build_city(fixed_plan_1) # <-- used for code development only, not marking
build_city(random_plan()) # <-- used for assessment

# Exit gracefully
# ***** Change the default argument to False if you want the
# ***** cursor (turtle) to remain visible at the end of the
# ***** program as a debugging aid
release_drawing_canvas()
