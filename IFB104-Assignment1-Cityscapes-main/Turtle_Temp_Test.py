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



import turtle


def buildingsite():       
    buildsite = plan[x][0] 
    if buildsite == (1) :
        print("1")
    elif buildsite == (2) :
        print ("2")
    elif buildsite == (3) :
        print ("3")
    elif buildsite == (4) :
        print ("4")
    elif buildsite == (5) :
        print ("5")
    elif buildsite == (6) :
        print ("6")
    elif buildsite == (7) :
        print ("7")
    elif buildsite == (8) :
        print ("8")
    elif buildsite == (9) :
        print ("9")
    elif buildsite == (10) :
        print ("10")

plan = city_plan 
        
def A_Base():
    t = turtle.Pen()
    t.reset()
    buildsite()
    t.up()
    t.goto(-165,0)
    t.down()
    t.width(10)
    t.color("blue")
    t.circle(75)    




from turtle import *
