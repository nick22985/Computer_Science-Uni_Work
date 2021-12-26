#-----Statement of Authorship----------------------------------------#
#
#  By submitting this task I agree that it represents my own work.
#  I am aware of the University rule that a student must not
#  act in a manner which constitutes academic dishonesty as stated
#  and explained in QUT's Manual of Policies and Procedures,
#  Section C/5.3 "Academic Integrity" and Section E/2.1 "Student
#  Code of Conduct".
#
#
#--------------------------------------------------------------------#

#----------------------------------------------------------------
#
# 1 label, 8 buttons (as per diagram)
# Initially, the title is empty.
# When the user hits one of the buttons, the title is changed (as per diagrams):
# When the user hits one of the Buttons, the title display changes depending on the Button:
# Monty Python Button changes the title to: "And now for something completely different..."
# The Actor1 Button changes the title to: "John Cleese".
# The Actor2 Button changes the title to: "Michael Palin".
# The Actor3 Button changes the title to: "Terry Gilliam".
# The Actor4 Button changes the title to: "Eric Idle, yes as in IDLE".
# The Actor5 Button changes the title to    : "Terry Jones".
# The Actor6 Button changes the title to: "Graham Chapman".
# The Clear Title Button clears the title (i.e., nothing is displayed in the title).

#----------------------------------------------------------------


# Import the Tkinter functions
from tkinter import *

# Create a window
the_window = Tk()

# Give the window a title
the_window.title('')

# PUT YOUR CODE HERE-------------------------------------------------#

def Monty_Python_Tag():                          
        the_window.title('And now for something completely different...')

def Actor1():                          
        the_window.title('John Cleese')
        
def Actor2():                          
        the_window.title('Michael Palin')

def Actor3():                          
        the_window.title('Terry Gilliam')

def Actor4():                          
        the_window.title('Eric Idle, yes as in IDLE')

def Actor5():                          
        the_window.title('Terry Jones')

def Actor6():                          
        the_window.title('Graham Chapman')

def Clear_Title():                          
        the_window.title('')

x='floralwhite'

frame1 = Frame(the_window, background=x)

label = Label(frame1, text='Set the Title',  background=x)
label.config(font=("Verdana 11 bold"))

button0 = Button(frame1, text = 'Monty Python Tag', background=x, command = Monty_Python_Tag)
button1 = Button(frame1, text = 'Actor1', background=x, command = Actor1)
button2 = Button(frame1, text = 'Actor2', background=x, command = Actor2)
button3 = Button(frame1, text = 'Actor3', background=x, command = Actor3)
button4 = Button(frame1, text = 'Actor4', background=x, command = Actor4)
button5 = Button(frame1, text = 'Actor5', background=x, command = Actor5)
button6 = Button(frame1, text = 'Actor6', background=x, command = Actor6)
button7 = Button(frame1, text = 'Clear Title', background='grey', command = Clear_Title)

the_window.configure(background=x)
the_window.geometry("430x87")
frame1.grid(row=2)
label.grid(row=0, column=3, columnspan=2, sticky=W+E)
button0.grid(row=2, column=1, padx=2, pady=5)
button1.grid(row=2, column=2, padx=2, pady=5)
button2.grid(row=2, column=3, padx=5, pady=5)
button3.grid(row=2, column=4, padx=5, pady=5)
button4.grid(row=2, column=5, padx=2, pady=5)
button5.grid(row=2, column=6, padx=2, pady=5)
button6.grid(row=2, column=7, padx=2, pady=5)
button7.grid(row=3, column=3, padx=2, pady=0, columnspan=2, sticky=W+E)




#----------------------------------------------------------------


# Start the event loop to react to user inputs
the_window.mainloop()
