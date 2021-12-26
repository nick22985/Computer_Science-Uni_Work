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
            else:                option = 'O'
            city_plan.append([site, style, num_floors, option])
    # Optionally print the result to the shell window
    if print_plan:
        print('\nBuildings to be constructed\n' +
              '(site, style, no. floors, option):\n\n',
              str(city_plan).replace('],', '],\n '))
    # Return the result to the student's build_city function
    return city_plan


def build_city(dummy_parameter):
    pass

build_city(random_plan())

#-----Logic to find out what site, style, height of the buildings----------------------#
import turtle
def build():
    Build_A()

#make this -10 building width everytime it is run for example else just do if b < 11:
#----------------------Turtle Building Template----------------------#
plan = city_plan 
from turtle import *

def Build_A():
    A_Base()
    B_Stories()
    A_Roof()

def Build_B():
    B_Base()
    B_Stories()
    B_Roof()    
    return

def Build_C():
    C_Base()
    C_Stories()
    C_Roof()
    return


def Build_D():
    D_Base()
    D_Stories()
    D_Roof()
    return
 
def A_Base():
    turtle.penup()
    turtle.pendown()
    turtle.forward(200)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(200)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    
def A_Stories():
    turtle.forward(100)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)
    turtle.forward(200)
    turtle.left(90)
    turtle.forward(50)
    turtle.left(90)

def A_Roof():
    return

def B_Base():
    return

def B_Stories():
    return

def B_Roof():
    return

def C_Base():
    return

def C_Stories():
    return

def C_Roof():
    return

def D_Base():
    return

def D_Stories():
    return

def D_Roof():
    return
    


       

