# -----Statement of Authorship----------------------------------------#
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
#  will not be marked.  Submitted files will be subjected to
#  software plagiarism analysis using the MoSS system
#  (http://theory.stanford.edu/~aiken/moss/).
#
# --------------------------------------------------------------------#
#
#
#
# -----Assignment Description-----------------------------------------#
#
#  Online Shopping Application
#
#  In this assignment you will combine your knowledge of HTMl/XML
#  mark-up languages with your skills in Python scripting, pattern
#  matching, and Graphical User Interface design to produce a useful
#  application for simulating an online shopping experience.  See
#  the instruction sheet accompanying this file for full details.
#
# --------------------------------------------------------------------#
#
#
#
# -----Imported Functions---------------------------------------------#
#
# Below are various import statements for helpful functions.  You
# should be able to complete this assignment using these
# functions only.  Note that not all of these functions are
# needed to successfully complete this assignment.
#
#
# The function for opening a web document given its URL.
# (You WILL need to use this function in your solution,
# either directly or via our "download" function.)
from urllib.request import urlopen

# Import the standard Tkinter functions. (You WILL need to use
# these functions in your solution.)
from tkinter import *

# Functions for finding all occurrences of a pattern
# defined via a regular expression, as well as
# the "multiline" and "dotall" flags.  (You do NOT need to
# use these functions in your solution, because the problem
# can be solved with the string "find" function, but it will
# be difficult to produce a concise and robust solution
# without using regular expressions.)
from re import findall, finditer, MULTILINE, DOTALL

# Import the standard SQLite functions (just in case they're
# needed).

import ctypes, os, webbrowser, random, sqlite3


#
# --------------------------------------------------------------------#
#
#
#
# -----Downloader Function--------------------------------------------#
#
# This is our function for downloading a web page's content and both
# saving it on a local file and returning its source code
# as a Unicode string. The function tries to produce
# a meaningful error message if the attempt fails.  WARNING: This
# function will silently overwrite the target file if it
# already exists!  NB: You should change the filename extension to
# "xhtml" when downloading an XML document.  (You do NOT need to use
# this function in your solution if you choose to call "urlopen"
# directly, but it is provided for your convenience.)
#


def download(url='https://www.amazon.com.au/gp/rss/bestsellers/electronics',
             target_filename='amazon_electronic_feed',
             filename_extension='html'):
    # Import an exception raised when a web server denies access
    # to a document
    from urllib.error import HTTPError

    # Open the web document for reading
    try:
        web_page = urlopen(url)
    except ValueError:
        raise Exception("Download error - Cannot find document at URL '" + url + "'")
    except HTTPError:
        raise Exception("Download error - Access denied to document at URL '" + url + "'")
    except:
        raise Exception("Download error - Something went wrong when trying to download " +
                        "the document at URL '" + url + "'")

    # Read its contents as a Unicode string
    try:
        web_page_contents = web_page.read().decode('UTF-8')
    except UnicodeDecodeError:
        raise Exception("Download error - Unable to decode document at URL '" +
                        url + "' as Unicode text")

    # Write the contents to a local text file as Unicode
    # characters (overwriting the file if it
    # already exists!)
    try:
        text_file = open(target_filename + '.' + filename_extension,
                         'w', encoding='UTF-8')
        text_file.write(web_page_contents)
        text_file.close()
    except:
        raise Exception("Download error - Unable to write to file '" +
                        target_filename + "'")

    # Return the downloaded document to the caller
    return web_page_contents


#
# --------------------------------------------------------------------#
# -----Student's Solution---------------------------------------------#
#
# Put your solution at the end of this file.
#
# Name of the invoice file. To simplify marking, your program should
# generate its invoice using this file name.

# -----Logic/Backend---------------------------------------------#

invoice_file = 'invoice.html'  # opens the invoice file and if it does not exist it will create it

abspath = os.path.abspath(__file__)  # finds where this file is
dname = os.path.dirname(abspath)  # shortcuting to this directory
os.chdir(dname)  # changing the directory to this directory


# -----Downloading RSS Feeds---------------------------------------------#
def download_rss():  # This function downloads RSS feeds
    global if_allowed  # setting global variables to acess them outside of this function
    global thinkgeek  # setting global variables to acess them outside of this function
    global fishing_tackle_shop  # setting global variables to acess them outside of this function
    global amazon_electronic_feed  # setting global variables to acess them outside of this function
    global ebay_pc  # setting global variables to acess them outside of this function
    # these are for the ofline copy's if you ever want them online
    # download('http://feeds.feedburner.com/thinkgeek/whatsnew', 'thinkgeek')
    # download('https://www.fishingtackleshop.com.au/rss.php?action=featuredproducts&type=rss', 'Fishing_Tackle_Shop')
    # download the updating rss feeds
    os.chdir("current")  # Changes directory to the folder called current in this folder
    # download('https://www.amazon.com.au/gp/rss/bestsellers/electronics',
    #        'amazon_electronic_feed')  # downlooads the rss feed from the amazon website and places it in the directory above. and names the file
    download('https://www.ebay.com.au/sch/i.html?&_nkw=pc&_rss=1',
             'pc-ebay')  # downlooads the rss feed from the ebay and places it in the directory above. and names the file
    os.chdir("..")  # makes the directory go back
    os.chdir("archived")  # changing directory to archived folder in this folder
    thinkgeek = open(
        'thinkgeek.html').read()  # this is opening the thinkgeek rss so the code can be acess within python
    fishing_tackle_shop = open(
        'Fishing_Tackle_Shop.html').read()  # this is opening the fishing tackle shops rss so the code can be acess within python
    os.chdir("..")  # makes the directory go back
    os.chdir("current")  # changing directory to the current folder in this folder
    amazon_electronic_feed = open(
        'amazon_electronic_feed.html').read()  # this is opening the amazon electronic feed rss so the code can be acess within python
    ebay_pc = open('pc-ebay.html').read()  # this is opening the pc ebay rss so the code can be acess within python
    os.chdir("..")  # makes the directory go back
    if_allowed()  # this is a function that is going to donwload the live data for the conversion from USD to AUD


def if_allowed(
        if_allowed=True):  # if you want the data to be static and the exchange rate not to change you just need to replace if_allowed(if_allowed = True) with if_allowed()
    global current_stock_usd_to_aud  # setting global variables to acess them outside of this function
    if if_allowed == True:  # Making a statment saying if this is true do this
        download('https://themoneyconverter.com/USD/AUD.aspx', 'stockexchange')  # downloads the website
        stockexchange = open('stockexchange.html').read()  # opens the website so it can be read in python
        currentstaticstockexchange = findall(
            '<h3>Quick Conversions from United States Dollar to Australian Dollar : 1 USD = \s*((?:.|\n)*?) AUD</h3>',
            stockexchange)  # find the infomation that is needed which is the conversion rate.
        current_stock_usd_to_aud = currentstaticstockexchange  # changing the variable of the price
        print(currentstaticstockexchange)
        os.remove("stockexchange.html")  # deleting the stockexchange file from directory
    else:  # making a statment of if none of the above fit do this
        current_stock_usd_to_aud = str(1.33)  # Setting the static value of the USD to AUD stock exxchange


# -----Making Blank List and variables---------------------------------------------#
# thinkgeek item list category.
cost_thinkgeek = []  # making a blank list for what the name is
category_thinkgeek = []  # making a blank list for what the name is
description_thinkgeek = []  # making a blank list for what the name is
product_thingeek = []  # making a blank list for what the name is
cart_thinkgeek = []  # making a blank list for what the name is
thinkgeek_y = 0  # setting a value to zero so we can acess it later
cartitem_thinkgeek = 1  # setting a value to 1 so we can acess it later

thinkgeek_category_checkout = []  # making blank checkout for said category
thinkgeek_description_checkout = []  # making blank checkout for said category
thinkgeek_cost_checkout = []  # making blank checkout for said category

# fishing_tackle_shop item list category.
cost_fishing_tackle_shop = []  # making a blank list for what the name is
category_fishing_tackle_shop = []  # making a blank list for what the name is
description_fishing_tackle_shop = []  # making a blank list for what the name is
product_fishing_tackle_shop = []  # making a blank list for what the name is
cart_fishing_tackle_shop = []  # making a blank list for what the name is
fishing_tackle_shop_y = 0  # setting a value to zero so we can acess it later
cartitem_fishing_tackle_shop = 0  # setting a value to zero so we can acess it later
image_fishing_tackle_shop = []  # making a blank list for what the name is

fishing_tackle_shop_category_checkout = []  # making blank checkout for said category
fishing_tackle_shop_description_checkout = []  # making blank checkout for said category
fishing_tackle_shop_cost_checkout = []  # making blank checkout for said category

# Amazon item list category.
cost_Amazon = []  # making a blank list for what the name is
image_Amazon = []  # making a blank list for what the name is
description_amazon_electronics = []  # making a blank list for what the name is
product_Amazon = []  # making a blank list for what the name is
product_amazon_electronics = []  # making a blank list for what the name is
length_Amazon = len(product_amazon_electronics) - 1  # setting a list length to a variable
amazon_number = 0  # setting a value to zero so we can acess it later
cartitem_amazon = 0  # setting a value to zero so we can acess it later
cart_amazon = []  # making a blank list for what the name is
amazon_y = 0  # setting a value to zero so we can acess it later

Amazon_category_checkout = []  # making blank checkout for said category
Amazon_description_checkout = []  # making blank checkout for said category
Amazon_cost_checkout = []  # making blank checkout for said category

# ebay_pc item list category.
cost_ebay_pc = []  # making a blank list for what the name is
category_ebay_pc = []  # making a blank list for what the name is
description_ebay_pc = []  # making a blank list for what the name is
product_ebay_pc = []  # making a blank list for what the name is
cart_ebay_pc = []  # making a blank list for what the name is
ebay_pc_y = 0  # setting a value to zero so we can acess it later
cartitem_ebay_pc = 0  # setting a value to zero so we can acess it later
image_ebay_pc = []  # making a blank list for what the name is

ebay_pc_category_checkout = []  # making blank checkout for said category
ebay_pc_description_checkout = []  # making blank checkout for said category
ebay_pc_cost_checkout = []  # making blank checkout for said category

y = 0  # making y = 0

name_description_price_final_cart = []  # making blank final checkout for said category
finallistcat = []  # making blank final list cat for said category
finallistdescription = []  # making blank final list des for said category
finallistcost = []  # making blank final list cost for said category
link = []  # making a blank list for what the name is
finallink = []  # making a blank list for what the name is
converted = []  # making a blank list for what the name is
pricethinkgeekconverted = []  # Making blank list to convert prices from US to AUD


# extracts data from the thijngeek webpage on the local machine
# -------------------------------------------------#

# -----Regular Expersions---------------------------------------------#
# thinkgeek regular expersion

def thinkgeek_full_item():
    global product_thingeek  # setting global variables to acess them outside of this function
    global category_thinkgeek  # setting global variables to acess them outside of this function
    global description_thinkgeek  # setting global variables to acess them outside of this function
    global cost_thinkgeek  # setting global variables to acess them outside of this function
    global pricethinkgeekconverted  # setting global variables to acess them outside of this function
    product_thingeek = findall('<item>\s*((?:.|\n)*?)</item>',
                               thinkgeek)  # using regular expersion to find two tages in the html file downloaded
    description_thinkgeek = findall("</guid>\s*((?:.|\n)*?)&lt;div",
                                    thinkgeek)  # using regular expersion to find two tages in the html file downloaded
    description_thinkgeek = [s.replace('<description>', '') for s in
                             description_thinkgeek]  # deleting <description> and replacing it with nothing in the listc
    description_thinkgeek = [s.replace('<em>', '') for s in
                             description_thinkgeek]  # deleting <em> and replacing it with nothing in the listc
    description_thinkgeek = [s.replace('</em>', '') for s in
                             description_thinkgeek]  # deleting </em> and replacing it with nothing in the listc
    category_thinkgeek = findall('Clothing :.*<',
                                 thinkgeek)  # using regular expersion to find two tages in the html file downloaded
    category_thinkgeek = [s.replace('<em>', '') for s in
                          category_thinkgeek]  # deleting < and replacing it with nothing in the listc
    category_thinkgeek = [s.replace('</em>', '') for s in
                          category_thinkgeek]  # deleting < and replacing it with nothing in the listc
    category_thinkgeek = [s.replace('<', '') for s in
                          category_thinkgeek]  # deleting < and replacing it with nothing in the listc
    category_thinkgeek = [s.replace(' &amp;', '') for s in
                          category_thinkgeek]  # deleting <description> and replacing it with nothing in the listc
    category_thinkgeek = [s.replace(' Statues', '') for s in
                          category_thinkgeek]  # deleting <description> and replacing it with nothing in the listc
    category_thinkgeek = [s.replace('Clothing', '') for s in
                          category_thinkgeek]  # deleting <description> and replacing it with nothing in the listc
    category_thinkgeek = [s.replace(' : ', '') for s in
                          category_thinkgeek]  # deleting <description> and replacing it with nothing in the listc
    cost_thinkgeek = findall('\$[\d]+.[\d]+',
                             thinkgeek)  # using regular expersion to find two tages in the html file downloaded
    lenthingeekcon = len(cost_thinkgeek) - 1  # setting a list length
    removesymbolforconvert = cost_thinkgeek  # making another var for list length so it cna be used later
    removesymbolforconvert = [s.replace('$', '') for s in removesymbolforconvert]  # removing $ symbol from said list
    f = 0  # making var f = 0
    while True:
        converstionrateUSD = 1 / float(
            current_stock_usd_to_aud[0])  # doing the conversion rate calculation from USD to AAUD
        cost_thingeek_converted = float(removesymbolforconvert[f]) / float(
            converstionrateUSD)  # doing the conversion rate calculation from USD to AAUD
        cost_thingeek_converted = str(round(cost_thingeek_converted, 2))  # rounding the convertion rate
        converted.append(str(cost_thingeek_converted))  # making conversion rate a string an adding it to another list
        f += 1  # stating f+1
        if f > lenthingeekcon:  # stating if f is greater then lenthingeekcon
            break  # then break. This avoids an infinite loop and also will allow us too make the statment to only repeat whatever the number of length thinkgeekcon is
    cconvertlen = len(converted) - 1  # turning another list length into a variablee for use latter
    k = 0  # making k = 0
    while True:  # stating while this statment is true
        templist = '$' + converted[k]  # this is making it so the list has a $ symbol at the start off it
        pricethinkgeekconverted.append(
            templist)  # this is adding the list that has just m,ade a dollar symbol and number into one big list
        k += 1  # making k + 1
        if k > cconvertlen:  # stating if k is greater then cconvertlen
            break  # then break. This avoids an infinite loop and also will allow us too make the statment to only repeat whatever the number of length cconvertlen is


# fishing tackle shop regular expersion
def fishing_tackle_shop_full_item():
    global product_fishing_tackle_shop  # setting global variables to acess them outside of this function
    global category_fishing_tackle_shop  # setting global variables to acess them outside of this function
    global description_fishing_tackle_shop  # setting global variables to acess them outside of this function
    global cost_fishing_tackle_shop  # setting global variables to acess them outside of this function
    global fishing_tackle_shop  # setting global variables to acess them outside of this function
    global image_fishing_tackle_shop  # setting global variables to acess them outside of this function
    product_fishing_tackle_shop = findall('<title>\s*((?:.|\n)*?)</title>',
                                          fishing_tackle_shop)  # using regular expersion to find two tages in the html file downloaded
    product_fishing_tackle_shop = [s.replace('<title>', '<title>#') for s in
                                   product_fishing_tackle_shop]  # adding the tag to have a hash after it so we can easily acess this later
    product_fishing_tackle_shop = [s.replace('<![CDATA[', '') for s in
                                   product_fishing_tackle_shop]  # remove said string and replacing with nothing
    product_fishing_tackle_shop = [s.replace(']]>', '') for s in
                                   product_fishing_tackle_shop]  # remove said string and replacing with nothing
    description_fishing_tackle_shop = findall("<p class=\"mainHeadingCustom\">\s*((?:.|\n)*?)</p>",
                                              fishing_tackle_shop)  # using regular expersion to find two tages in the html file downloaded
    description_fishing_tackle_shop = [s.replace('"mainHeadingCustom">', '') for s in
                                       description_fishing_tackle_shop]  # remove said string and replacing with nothing
    cost_fishing_tackle_shop = findall('<isc:price>\s*((?:.|\n)*?)</isc:price>+',
                                       fishing_tackle_shop)  # using regular expersion to find two tages in the html file downloaded
    cost_fishing_tackle_shop = [s.replace('<isc:price><![CDATA[', '') for s in
                                cost_fishing_tackle_shop]  # remove said string and replacing with nothing
    cost_fishing_tackle_shop = [s.replace(']]></isc:price>', '') for s in
                                cost_fishing_tackle_shop]  # remove said string and replacing with nothing
    cost_fishing_tackle_shop = [s.replace('<![CDATA[', '') for s in
                                cost_fishing_tackle_shop]  # remove said string and replacing with nothing
    cost_fishing_tackle_shop = [s.replace(']]>', '') for s in
                                cost_fishing_tackle_shop]  # remove said string and replacing with nothing
    del product_fishing_tackle_shop[0]  # deleting list item number 0 from the list. This is because it is usless to us

    kk = 1
    for i in range(10):
        del description_fishing_tackle_shop[kk]
        del description_fishing_tackle_shop[kk]
        kk += 1
    print(description_fishing_tackle_shop)


# amazon regualar experions
def amazon_full_item():
    global product_amazon_electronics  # setting global variables to acess them outside of this function
    global description_amazon_electronics  # setting global variables to acess them outside of this function
    global cost_amazon_electronics  # setting global variables to acess them outside of this function
    global length_Amazon  # setting global variables to acess them outside of this function
    global amazon_number  # setting global variables to acess them outside of this function
    global amazon_number  # setting global variables to acess them outside of this function
    global image_Amazon  # setting global variables to acess them outside of this function
    global amazon_electronic_feed  # setting global variables to acess them outside of this function
    t = 0  # setting var t to 0
    product_amazon_electronics = findall('<title>#\s*((?:.|\n)*?)</title>',
                                         amazon_electronic_feed)  # using regular expersion to find two tages in the html file downloaded
    product_amazon_electronics = [s.replace('<title>', '<title>#') for s in
                                  product_amazon_electronics]  # adding the tag to have a hash after it so we can easily acess this later
    product_amazon_electronics = [s.replace('6&quot;', '') for s in
                                  product_amazon_electronics]  # remove said string and replacing with nothing
    product_amazon_electronics = [s.replace(' aptx ', '') for s in
                                  product_amazon_electronics]  # remove said string and replacing with nothing
    product_amazon_electronics = [s.replace('â€“ ', '-') for s in
                                  product_amazon_electronics]  # remove said string and replacing with nothing
    for i in range(10):  # stating for I in range 10 means it will repeat 10 times
        t += 1  # making t + 1
        amazon_number = t  # making amazon_number = t so the loop is not infinite if t changes
        amazon_number = f'{amazon_number}: '  # making it so i can directly insert a variable into the striong
        product_amazon_electronics = [s.replace(amazon_number, '') for s in
                                      product_amazon_electronics]  # remove said string and replacing with nothing
    first_amazon_image = findall('<img src="https://images-fe.ssl-images\s*((?:.|\n)*?)"',
                           amazon_electronic_feed)  # using regular expersion to find two tages in the html file downloaded

    lengthofrepeat = len(first_amazon_image) - 1
    z = 0
    for i in range(lengthofrepeat):
        temp_list1 = 'https://images-fe.ssl-images' + first_amazon_image[z]
        image_Amazon.append(temp_list1)
        z += 1

    cost_amazon_electronics = findall('\$[\d]+.[\d]+',
                                      amazon_electronic_feed)  # using regular expersion to find two tages in the html file downloaded
    print(image_Amazon)


# ebay regular expersion
def ebay_pc_full_item():
    global product_ebay_pc  # setting global variables to acess them outside of this function
    global category_ebay_pc  # setting global variables to acess them outside of this function
    global description_ebay_pc  # setting global variables to acess them outside of this function
    global cost_ebay_pc  # setting global variables to acess them outside of this function
    global image_ebay_pc  # setting global variables to acess them outside of this function
    global ebay_pc  # setting global variables to acess them outside of this function
    product_ebay_pc = findall('<title>\s*((?:.|\n)*?)</title>',
                              ebay_pc)  # using regular expersion to find two tages in the html file downloaded
    product_ebay_pc = [s.replace('<title>', '<title>#') for s in
                       product_ebay_pc]  # adding the tag to have a hash after it so we can easily acess this later
    product_ebay_pc = [s.replace('<![CDATA[pc]]>', '') for s in
                       product_ebay_pc]  # remove said string and replacing with nothing
    product_ebay_pc = [s.replace('<![CDATA[', '') for s in
                       product_ebay_pc]  # remove said string and replacing with nothing
    product_ebay_pc = [s.replace(']]', '') for s in product_ebay_pc]  # remove said string and replacing with nothing
    product_ebay_pc = [s.replace('>', '') for s in product_ebay_pc]  # remove said string and replacing with nothing
    cost_ebay_pc = findall('<b>AU \$</b>\s*((?:.|\n)*?)<',
                           ebay_pc)  # using regular expersion to find two tages in the html file downloaded
    image_ebay_pc = findall('src=\s*((?:.|\n)*?)>',
                            ebay_pc)  # using regular expersion to find two tages in the html file downloaded
    image_ebay_pc = [s.replace('"', '') for s in
                     image_ebay_pc]  # remove said string and replacing with nothing]  #remove said string and replacing with nothing
    del product_ebay_pc[0]  # deleting item nb 0 from list
    del product_ebay_pc[0]  # deleting item nb 0 from list
    a = 0  # making a = 0
    for i in range(10):  # stating for I in range 10 means it will repeat 10 times
        del cost_ebay_pc[a]  # telling the list to delete list item a from ebay
        a += 1  # making a + 1 to change the number of what the item is deleting


# -----Setting Variabels---------------------------------------------#
# setting thinkgeeks variables
def thinkgeek_var():
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    thinkgeek_full_item()  # running the function
    category = category_thinkgeek  # making category = said item from it regulare experion
    description = description_thinkgeek  # making description = said item from it regular expression
    cost = pricethinkgeekconverted  # making description = said item from it regular expression
    listcategory = thinkgeek_category_checkout  # making description = said item from a saif variable
    listdescription = thinkgeek_description_checkout  # making description = said item from a saif variable
    listcost = thinkgeek_cost_checkout  # making description = said item from a saif variable
    cart_type = cart_thinkgeek  # making description = said item from a saif variable


# setting fishing tackle shop variables
def fishing_tackle_shop_var():
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    fishing_tackle_shop_full_item()  # running the function
    category = product_fishing_tackle_shop  # making category = said item from it regulare experion
    description = description_fishing_tackle_shop  # making description = said item from it regular expression
    cost = cost_fishing_tackle_shop  # making description = said item from it regular expression
    listcategory = fishing_tackle_shop_category_checkout  # making description = said item from a saif variable
    listdescription = fishing_tackle_shop_description_checkout  # making description = said item from a saif variable
    listcost = fishing_tackle_shop_cost_checkout  # making description = said item from a saif variable
    cart_type = cart_fishing_tackle_shop  # making cart = said item from a saif variable


# Setting Amazon Variables
def amazon_var():
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    global link  # setting global variables to acess them outside of this function
    amazon_full_item()  # running the function
    category = product_amazon_electronics  # making category = said item from it regulare experion
    description = image_Amazon  # making description = said item from it regular expression
    cost = cost_amazon_electronics  # making description = said item from it regular expression
    listcategory = Amazon_category_checkout  # making description = said item from a saif variable
    listdescription = Amazon_description_checkout  # making description = said item from a saif variable
    listcost = Amazon_cost_checkout  # making description = said item from a saif variable
    cart_type = cart_amazon  # making cart = said item from a saif variable


# setting ebay variables
def ebay_pc_var():
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    ebay_pc_full_item()  # running the function
    category = product_ebay_pc  # making category = said item from it regulare experion
    description = image_ebay_pc  # making description = said item from it regular expression
    cost = cost_ebay_pc  # making description = said item from a saif variable
    listcategory = ebay_pc_category_checkout  # making description = said item from a saif variable
    listdescription = ebay_pc_description_checkout  # making description = said item from a saif variable
    listcost = ebay_pc_cost_checkout  # making description = said item from a saif variable
    cart_type = cart_ebay_pc  # making cart = said item from a saif variable


# -----Joing list together to start making final carts---------------------------------------------#

def prices():
    global cartnumber  # setting global variables to acess them outside of this function
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    global finallistcat  # setting global variables to acess them outside of this function
    global finallistdescription  # setting global variables to acess them outside of this function
    global finallistcost  # setting global variables to acess them outside of this function
    cartnumber = cart_type[y]  # making cartnumber  = cart type [y]. Y changes so it will go up in the list
    listcategory.append(
        category[cartnumber])  # making is so that the items from the said category will go into the final cart
    listdescription.append(
        description[cartnumber])  # making is so that the items from the said category will go into the final cart
    listcost.append(cost[cartnumber])  # making is so that the items from the said category will go into the final cart
    finallistcat.append(
        category[cartnumber])  # making is so that the items from the said category will go into the final cart
    finallistdescription.append(
        description[cartnumber])  # making is so that the items from the said category will go into the final cart
    finallistcost.append(
        cost[cartnumber])  # making is so that the items from the said category will go into the final cart
    print(listcategory)  # printing said category in console for debuging purposes
    print(listdescription)  # printing said category in console for debuging purposes
    print(listcost)  # printing said category in console for debuging purposes
    list_ordering()  # making the list in to one giant list


# joing all the list together to mae it one giant list so that you can acess it all from one list instead of 3
def list_ordering():
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global cost  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function
    global name_description_price_final_cart  # setting global variables to acess them outside of this function
    global cartnumber  # setting global variables to acess them outside of this function
    temp1 = []  # mnaking a blank temp list
    temp1.append(category[cartnumber])  # making said category join with temp 1
    temp1.append(description[cartnumber])  # making said category join with temp 1
    temp1.append(cost[cartnumber])  # making said category join with temp 1
    name_description_price_final_cart.append(
        temp1)  # making the list temp1 into name description so we can start the next list and it will not get over writed because appedning is adding onto the list
    print(
        "-----------------------------------------------------------------------------------------------------------")  # purly for debuging purposes in console so it is seperated from tother things
    print(name_description_price_final_cart)  # prints out said list
    print(
        "-----------------------------------------------------------------------------------------------------------")  # purly for debuging purposes in console so it is seperated from tother things


# going throught the cart numbers to convert is to prices
def cart_numbering():
    global cartnumber  # setting global variables to acess them outside of this function
    global category  # setting global variables to acess them outside of this function
    global description  # setting global variables to acess them outside of this function
    global listcategory  # setting global variables to acess them outside of this function
    global listdescription  # setting global variables to acess them outside of this function
    global listcost  # setting global variables to acess them outside of this function
    global cart_type  # setting global variables to acess them outside of this function
    global length_of_cart  # setting global variables to acess them outside of this function
    global y  # setting global variables to acess them outside of this function

    length_of_cart = len(cart_type) - 1  # setting the a variable to the length of the cart type
    while True:
        prices()  # running this function
        y += 1  # making y + 1
        if y > length_of_cart:  # making it so that it will only repeat y amount of times
            y = 0  # reseting y
            break  # making the loop break


# making the function to add all the cost together to get a total cost for the invoice
def total_cost():
    global totalcost  # setting global variables to acess them outside of this function
    totalcost = finallistcost  # making final cost into another variable so we do not overide final cost
    totalcost = [s.replace('$', '') for s in totalcost]  # replacing all $ in total cost
    totalcost = [s.replace(',', '') for s in totalcost]  # replacing all , in total cost
    totalcost = [round(float(i), 2) for i in
                 totalcost]  # making total cost repeat x amount and making it to two decimal places
    totalcost = sum(map(int, totalcost))  # adding up total cost and rounding


# -------------------------------------------------------#
# -------------------------------------------------------#
# --------------------------GUI---------------------------#
#
# --------------------Variables For GUI Popout Window---------------------------#
# setting shop 4 variables
def shop_4_var():
    global frametitle  # setting global variables to acess them outside of this function
    global fg  # setting global variables to acess them outside of this function
    global bg  # setting global variables to acess them outside of this function
    global itembg  # setting global variables to acess them outside of this function
    global itemfg  # setting global variables to acess them outside of this function
    global font  # setting global variables to acess them outside of this function
    global website  # setting global variables to acess them outside of this function
    global storelistype  # setting global variables to acess them outside of this function
    global costlisttype  # setting global variables to acess them outside of this function
    frametitle = 'Desktop PC'  # setting frame title
    bg = 'black'  # setting background colour
    fg = 'white'  # setting foreground colour
    itembg = 'white'  # setting item background colour
    itemfg = 'black'  # setting item foreground colour
    font = 5  # setting font size
    website = 'https://www.ebay.com.au/sch/i.html?&_nkw=pc&_rss=1'  # setting the website
    storelistype = product_ebay_pc  # setting the store listtype
    costlisttype = cost_ebay_pc  # setting the costlist type


# setting shop 3 variables
def shop_3_var():
    global frametitle  # setting global variables to acess them outside of this function
    global fg  # setting global variables to acess them outside of this function
    global bg  # setting global variables to acess them outside of this function
    global itembg  # setting global variables to acess them outside of this function
    global itemfg  # setting global variables to acess them outside of this function
    global font  # setting global variables to acess them outside of this function
    global website  # setting global variables to acess them outside of this function
    global storelistype  # setting global variables to acess them outside of this function
    global costlisttype  # setting global variables to acess them outside of this function
    frametitle = 'Electronics'  # setting the frame title
    bg = 'green'  # setting background colour
    fg = 'white'  # setting foreground colour
    itembg = 'white'  # setting item background colour
    itemfg = 'black'  # setting item foreground colour
    font = 5  # setting font
    website = 'https://www.amazon.com.au/gp/rss/bestsellers/electronics'  # setting the website
    storelistype = product_amazon_electronics  # settijng the store list type
    costlisttype = cost_amazon_electronics  # setting the costlistype


# setting shop 2 variables
def shop_2_var():
    global frametitle  # setting global variables to acess them outside of this function
    global fg  # setting global variables to acess them outside of this function
    global bg  # setting global variables to acess them outside of this function
    global itembg  # setting global variables to acess them outside of this function
    global itemfg  # setting global variables to acess them outside of this function
    global font  # setting global variables to acess them outside of this function
    global website  # setting global variables to acess them outside of this function
    global storelistype  # setting global variables to acess them outside of this function
    global costlisttype  # setting global variables to acess them outside of this function
    frametitle = 'Fishing Gear'  # setting the store title
    bg = 'darkred'  # setting the background colour
    fg = 'white'  # setting the foreground colour
    itembg = 'white'  # setting the item background colour
    itemfg = 'black'  # settiung the item foreground colour
    font = 5  # setting the font size
    website = 'https://goo.gl/TQ4GVH'  # setting the website
    storelistype = product_fishing_tackle_shop  # setting the storelistype
    costlisttype = cost_fishing_tackle_shop  # setting the cost list type


# setting shop 1 variables
def shop_1_var():
    global frametitle  # setting global variables to acess them outside of this function
    global fg  # setting global variables to acess them outside of this function
    global bg  # setting global variables to acess them outside of this function
    global itembg  # setting global variables to acess them outside of this function
    global itemfg  # setting global variables to acess them outside of this function
    global font  # setting global variables to acess them outside of this function
    global website  # setting global variables to acess them outside of this function
    global storelistype  # setting global variables to acess them outside of this function
    global costlisttype  # setting global variables to acess them outside of this function
    frametitle = 'Clothing'  # setting the shop title
    bg = 'darkblue'  # setting the background colour
    fg = 'white'  # setting the foregrounf colour
    itembg = 'white'  # setting item background colour
    itemfg = 'black'  # setting ite m foregrounf colour
    font = 5  # setting the font size
    website = 'http://feeds.feedburner.com/thinkgeek/whatsnew'  # setting the website
    storelistype = category_thinkgeek  # setting the store list type
    costlisttype = cost_thinkgeek  # setting the costlisttype


cartlistforcart = []  # setting blank list
hash_list = []  # setting blank list


def popupwindows():
    global root  # setting global variables to acess them outside of this function
    global category_thinkgeek  # setting global variables to acess them outside of this function
    global frametitle  # setting global variables to acess them outside of this function
    global fg  # setting global variables to acess them outside of this function
    global bg  # setting global variables to acess them outside of this function
    global itembg  # setting global variables to acess them outside of this function
    global itemfg  # setting global variables to acess them outside of this function
    global font  # setting global variables to acess them outside of this function
    global website  # setting global variables to acess them outside of this function
    global storelistype  # setting global variables to acess them outside of this function
    global costlisttype  # setting global variables to acess them outside of this function
    global master  # setting global variables to acess them outside of this function
    global list_10  # setting global variables to acess them outside of this function
    list_10 = ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9',
               '#10']  # setting the list for what nb the product is
    # Create a window
    master = Toplevel(root)  # making the window toplevel so that it destroy itself if the root window is closed
    master.title(frametitle)  # setting the master title to the frame title var
    #   changeable perimeters
    #   frames
    frame1 = Frame(master, bg='black')  # making frame 1
    frame2 = Frame(master, bg="black")  # making Frame 2
    itemlist = Frame(master, bg=itembg, highlightbackground="black", highlightcolor="black", highlightthickness=1,
                     bd=0)  # making itemlist frame
    #   configuration for master window
    master.configure(background=bg)  # changing config of background on window to var bg
    #   Label's
    Label(frame1, text=frametitle, font=1, justify=LEFT, anchor=W, bg=bg, fg=fg).grid(row=0, column=0,
                                                                                      sticky=W)  # setting main title of the frame to var frametitle
    #   ItemNumber
    nb = Label(itemlist, text='Nb.', bg=itembg)  # priting out nb
    nb.config(anchor=W, justify=LEFT, fg='red', font=font)  # changing config of nbk
    itemnumber1 = Label(itemlist, text=("".join(list_10[0])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber2 = Label(itemlist, text=("".join(list_10[1])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber3 = Label(itemlist, text=("".join(list_10[2])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber4 = Label(itemlist, text=("".join(list_10[3])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber5 = Label(itemlist, text=("".join(list_10[4])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber6 = Label(itemlist, text=("".join(list_10[5])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber7 = Label(itemlist, text=("".join(list_10[6])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber8 = Label(itemlist, text=("".join(list_10[7])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber9 = Label(itemlist, text=("".join(list_10[8])), fg='red', bg=itembg,
                        font=font)  # joining said list with said itemnumber and configuring
    itemnumber10 = Label(itemlist, text=("".join(list_10[9])), fg='red', bg=itembg,
                         font=font)  # joining said list with said itemnumber and configuring

    # ------------------------------------------------------- #
    #   Name
    item = Label(itemlist, text='Item', justify=LEFT, font=font, anchor=W, bg=itembg,
                 fg=itemfg)  # making it print out item
    itemnames1 = Label(itemlist, text=("".join(storelistype[0])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames2 = Label(itemlist, text=("".join(storelistype[1])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames3 = Label(itemlist, text=("".join(storelistype[2])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames4 = Label(itemlist, text=("".join(storelistype[3])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames5 = Label(itemlist, text=("".join(storelistype[4])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames6 = Label(itemlist, text=("".join(storelistype[5])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames7 = Label(itemlist, text=("".join(storelistype[6])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames8 = Label(itemlist, text=("".join(storelistype[7])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames9 = Label(itemlist, text=("".join(storelistype[8])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                       wraplengt=330)  # joining said list with said itemnumber and configuring
    itemnames10 = Label(itemlist, text=("".join(storelistype[9])), justify=LEFT, font=font, bg=itembg, fg=itemfg,
                        wraplengt=330)  # joining said list with said itemnumber and configuring
    # ------------------------------------------------------- #
    #   Item Price
    cost1 = Label(itemlist, text='Price', anchor=W, justify=LEFT, bg=itembg, fg='green',
                  font=font)  # making the title Price
    price1 = Label(itemlist, text=("".join(costlisttype[0])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price2 = Label(itemlist, text=("".join(costlisttype[1])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price3 = Label(itemlist, text=("".join(costlisttype[2])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price4 = Label(itemlist, text=("".join(costlisttype[3])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price5 = Label(itemlist, text=("".join(costlisttype[4])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price6 = Label(itemlist, text=("".join(costlisttype[5])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price7 = Label(itemlist, text=("".join(costlisttype[6])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price8 = Label(itemlist, text=("".join(costlisttype[7])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price9 = Label(itemlist, text=("".join(costlisttype[8])), justify=LEFT, fg='green', font=font,
                   bg=itembg)  # joining said list with said itemnumber and configuring
    price10 = Label(itemlist, text=("".join(costlisttype[9])), justify=LEFT, fg='green', font=font,
                    bg=itembg)  # joining said list with said itemnumber and configuring
    websiteurl = Label(frame2, text=website, padx='5', bg=bg, fg=fg)  # setting the website URL
    # ------------------------------------------------------- #
    #   Item placement
    frame1.grid(row=0,
                sticky=N)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    nb.grid(row=0, column=0,
            sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    item.grid(row=0, column=1,
              sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    cost1.grid(row=0, column=2,
               sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber1.grid(row=1, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber2.grid(row=2, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber3.grid(row=3, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber4.grid(row=4, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber5.grid(row=5, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber6.grid(row=6, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber7.grid(row=7, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber8.grid(row=8, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber9.grid(row=9, column=0,
                     sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    itemnumber10.grid(row=10, column=0, sticky=W)
    itemlist.grid(row=1, column=0, sticky=W, padx=(10,
                                                   10))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames1.grid(row=1, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames2.grid(row=2, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames3.grid(row=3, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames4.grid(row=4, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames5.grid(row=5, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames6.grid(row=6, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames7.grid(row=7, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames8.grid(row=8, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames9.grid(row=9, column=1, sticky=W, pady=(5,
                                                     5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    itemnames10.grid(row=10, column=1, sticky=W, pady=(5,
                                                       5))  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created also setting padding
    price1.grid(row=1, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price2.grid(row=2, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price3.grid(row=3, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price4.grid(row=4, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price5.grid(row=5, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price6.grid(row=6, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price7.grid(row=7, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price8.grid(row=8, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price9.grid(row=9, column=2,
                sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    price10.grid(row=10, column=2,
                 sticky=W)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    frame2.grid(row=2,
                sticky=N)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created
    websiteurl.grid(row=0,
                    column=2)  # placing said thing in said row/colum and also setting where it is going to stick to or go towards when created


# making function to destroy master window
def destroy_window():
    master.destroy()  # this will destroy master window


# defining selection
def sel():
    global master  # setting global variables to acess them outside of this function
    global var  # setting global variables to acess them outside of this function
    selection = str(var.get())  # setting selection to the string of var.get which is getting the variable of var
    if selection == '1':  # saying if selection is equal to 1
        thinkgeek_full_item()  # running said function
        destroy_window()  # running the destroy function to delete the popup windows os their is only one when you click other buttons
        shop_1_var()  # setting variables
        popupwindows()  # Running the popup window
    elif selection == '2':  # saying if selection is equal to 2
        fishing_tackle_shop_full_item()  # running said function
        destroy_window()  # running the destroy function to delete the popup windows os their is only one when you click other buttons
        shop_2_var()  # setting variables
        popupwindows()  # Running the popup window
    elif selection == '3':  # saying if selection is equal to 3
        amazon_full_item()  # running said function
        destroy_window()  # running the destroy function to delete the popup windows os their is only one when you click other buttons
        shop_3_var()  # setting variables
        popupwindows()  # Running the popup window
    elif selection == '4':  # saying if selection is equal to 4
        ebay_pc_full_item()  # running said function
        destroy_window() # running the destroy function to delete the popup windows os their is only one when you click other buttons
        shop_4_var() # setting variables
        popupwindows() # Running the popup window

#defining how to add suff to the cvorrect list / cart type
def cart_item():
    global cart_thinkgeek # setting global variables to acess them outside of this function
    global cart_fishing_tackle_shop # setting global variables to acess them outside of this function
    global cart_amazon # setting global variables to acess them outside of this function
    global cart_ebay_pc # setting global variables to acess them outside of this function
    selection = str(var.get()) # setting selection to str var get
    number = int(variable.get()) # setting number to variable .get number
    number -= 1 #making number - 1
    if selection == '0':  # saying if selection is equal to 0
        ctypes.windll.user32.MessageBoxW(0, "ERROR: You have not selected a store",
                                         "Error: Adding to Cart", 0)
    elif selection == '1': # saying if selection is equal to 1
        cart_thinkgeek.append(number) #making it so that it adds whatver number number euquals to the cart and if the selection is equal to 1
        print(cart_thinkgeek) #prints said variable
    elif selection == '2': # saying if selection is equal to 2
        cart_fishing_tackle_shop.append(number) #making it so that it adds whatver number number euquals to the cart and if the selection is equal to 2
        print(cart_fishing_tackle_shop) #prints said variable
    elif selection == '3': # saying if selection is equal to 3
        cart_amazon.append(number) #making it so that it adds whatver number number euquals to the cart and if the selection is equal to 3
        print(cart_amazon) #prints said variable
    elif selection == '4': # saying if selection is equal to 4
        cart_ebay_pc.append(number) #making it so that it adds whatver number number euquals to the cart and if the selection is equal to 4
        print(cart_ebay_pc) #prints said variable

#definging cart length
def cartlength():
    global cat # setting global variables to acess them outside of this function
    global des # setting global variables to acess them outside of this function
    global cos # setting global variables to acess them outside of this function
    global img # setting global variables to acess them outside of this function
    global dblist # setting global variables to acess them outside of this function
    g = 0 #setting g to -
    lengthoffinal = len(finallistcat) - 1 #setting var lengthoffinal to len of the final cart - 1
    while True:
        dblist = [] #making blank list
        cat = finallistcat[g] # making it so that cet = the said list and var
        des = finallistdescription[g] # making it so that des = the said list and var
        cos = finallistcost[g] # making it so that cos = the said list and var
        writeitemdes() #run said function
        dblist.append(cat) #appending cat to dblist
        dblist.append(cos) #appending cos to dblist
        data_entry() #running said funcition
        g += 1 #making g + 1
        if g > lengthoffinal: #making it so that if g is greater than length of final then break
            break

#database making
def data_entry():
    conn = sqlite3.connect('shopping_cart.db') #opens the database
    c = conn.cursor() #makes it so that you can edit the data base
    c.execute('insert into ShoppingCart values (?,?)', dblist) # telling the datablesa to insert Into Shopping cart whatever 2 values dblist is
    conn.commit() # this is saving changes to the database

a = 0 #making a = 0

#This function writes the item in HTML
def writeitemdes():
    global a # setting global variables to acess them outside of this function
    a += 1 #making a + 1
    global invoice_file # setting global variables to acess them outside of this function
    invoice_file.write('<div class="itemarea">') #telling what to write in the invoice file html
    invoice_file.write('<h4>' + str(cat) + '</h4>\n') #telling what to write in the invoice file html

    if des.startswith('http'): #saying if des starts with HTTP do this
        invoice_file.write('<img src= "' + str(des) + '"alt = "cfg">\n') #this puts the src tage so the image is displayed as an image
    else: #saying if anything else do this
        invoice_file.write('<p>' + str(des) + '</p>\n') #which is telling it to put the paragrappth tag to make a paragrpath

    invoice_file.write('<p> Our Price:' + str(cos) + 'AUD</p>\n </div> ') #telling the html file top write what is in the description

#setting invoice number
def random_invoice_number():
    global ran # setting global variables to acess them outside of this function
    ran = random.randint(1, 9999) #making it so that it chooses a random number beetween 1 and 9999

#Making the invoice template
def invoice_template():
    global cartlengthfinal # setting global variables to acess them outside of this function
    global invoice_file # setting global variables to acess them outside of this function
    random_invoice_number() #running said function
    # Open the target HTML file (sf_movies.html) for writing as Unicode
    invoice_file = open('invoice.html', 'w', encoding='UTF-8')

    # Write standard HTML "header" markups into your file, up to
    # and including the <body> tag.  Give your web document a
    # meaningful title
    invoice_file.write('''<!DOCTYPE html>
    <html>
      <head>
      <link rel="shortcut icon" type="image/x-icon" href="just_logo.ico" />
      </head>
      <style>
      h1 {text-align:center}
      h2 {text-align:center}
      h3 {text-align:left}

    .contentarea{
	    width: 40%;
	    margin-left: auto;
        margin-right: auto;
	    padding: 5px;
	    text-align: center;
	    border-style: ridge;
	    }
	.itemarea{
	    width: 95%;
	    margin-left: auto;
        margin-right: auto;
	    text-align: center;
	    border-style: ridge;
	     }
      </style>
      <body>
      <title>Offline - Invoice</title>
      <h1><img src="https://i.imgur.com/xddj5ZU.png"></h1>
    ''')
    invoice_file.write('<div class="contentarea">') #writing to the invoice file what is in the ()

    invoice_file.write('<h3>Invoice #' + str(ran) + '</h3>\n') #writing the random invoice number to the file

    cartlength() #running said function

    invoice_file.write('<h2>Total for this purchase: $') #writing said thing
    invoice_file.write(str(totalcost)) # writing the said list
    invoice_file.write(''' AUD </h2> \n \n
    <p>Please pay as soon as possible so we can ship your order. \n
    The payment Gateways that we accept are <b><i>Mastercard</i></b> and <b><i>Paypal</i></b>. \n
    <br><b><i>Thank you for shopping with Offline.</i></b></br></p>
    ''') #writing the said thing

    invoice_file.write('''</div> \n  </body> \n
    </html>''') #ending the html document open tags

    # Close your HTML file (which you can now view in a web browser)
    invoice_file.close()

#cleaning up said list so you can repeatly add to the invoice as long as the application does not close
def cleanup():
    global cart_thinkgeek # setting global variables to acess them outside of this function
    global cart_fishing_tackle_shop # setting global variables to acess them outside of this function
    global cart_amazon # setting global variables to acess them outside of this function
    global cart_ebay_pc # setting global variables to acess them outside of this function
    # cleanup so if user want they can add different items to the cart
    cart_thinkgeek = [] #making blank list
    cart_fishing_tackle_shop = [] #making blank list
    cart_amazon = [] #making blank list
    cart_ebay_pc = [] #making blank list

#making if then statment to convert the numbers that this is given and put it inot text
def print_invoice():
    global var # setting global variables to acess them outside of this function
    selection = str(var.get()) #getting var get
    if len(cart_thinkgeek) > 0: #saying if cart thingeek has something in it do this
        thinkgeek_var() #running said function
        cart_numbering() #running said function
        if len(cart_fishing_tackle_shop) > 0: #saying if cart fishing has something in it do this
            fishing_tackle_shop_var() #running said function
            cart_numbering() #running said function
            if len(cart_amazon) > 0: #saying if cart amazon has something in it do this
                amazon_var() #running said function
                cart_numbering() #running said function
                if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
                    ebay_pc_var()  #running said function
                    cart_numbering()  #running said function
                else: #telling it if none of above
                    pass #telling the statment to pass and move onto the next elif/else statment
            else: #telling it if none of above
                pass #telling the statment to pass and move onto the next elif/else statment
        else: #telling it if none of above
            if len(cart_amazon) > 0: #saying if cart amazon has something in it do this
                amazon_var()  #running said function
                cart_numbering()  #running said function
                if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
                    ebay_pc_var()  #running said function
                    cart_numbering()  #running said function
                else: #telling it if none of above
                    if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
                        ebay_pc_var()  #running said function
                        cart_numbering()  #running said function
                    else: #telling it if none of above
                        pass #telling the statment to pass and move onto the next elif/else statment
            else: #telling it if none of above
                pass #telling the statment to pass and move onto the next elif/else statment
    elif len(cart_fishing_tackle_shop) > 0: #saying if cart fishing has something in it do this
        fishing_tackle_shop_var()  #running said function
        cart_numbering()  #running said function
        if len(cart_amazon) > 0: #saying if cart amazon has something in it do this
            amazon_var()  #running said function
            cart_numbering()  #running said function
            if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
                ebay_pc_var()  #running said function
                cart_numbering()  #running said function
            else: #telling it if none of above
                pass #telling the statment to pass and move onto the next elif/else statment
        else: #telling it if none of above
            if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
                ebay_pc_var()  #running said function
                cart_numbering()  #running said function
            else: #telling it if none of above
                pass #telling the statment to pass and move onto the next elif/else statment
    elif len(cart_amazon) > 0: #saying if cart amazon has something in it do this
        amazon_var()  #running said function
        cart_numbering()  #running said function
        if len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
            ebay_pc_var()  #running said function
            cart_numbering()  #running said function
        else:  #telling it if none of above
            pass #telling the statment to pass and move onto the next elif/else statment
    elif len(cart_ebay_pc) > 0: #saying if cart ebay has something in it do this
        ebay_pc_var()  #running said function
        cart_numbering()  #running said function

    elif selection == '0': #saying if selection = 0
        ctypes.windll.user32.MessageBoxW(0, "ERROR: You have not selected a store",
                                         "Error: Creating Invoice", 0) #this is making a popup box tell you what is wrong
    else: #telling it if none of above
        ctypes.windll.user32.MessageBoxW(0, "ERROR: You have not added any items into your cart",
                                         "Error: Creating Invoice", 0) #this is making a popup box tell you what is wrong
    total_cost()  #running said function
    invoice_template()  #running said function
    cleanup()  #running said function


def center_window(width=300, height=200): # setting window perimiters
    # get screen width and height
    screen_width = root.winfo_screenwidth() #getting screen width
    screen_height = root.winfo_screenheight() #getting screen height

    # calculate position x and y coordinates
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

#making the open invoice button
def open_invoice():
    file = dname + "\\invoice.html" #stating where the file is located
    if os.path.exists(file) == True:
        webbrowser.open('file://' + os.path.realpath('invoice.html')) #Telling the webbrowser to open said path
    else:
        ctypes.windll.user32.MessageBoxW(0, "ERROR: No Invoice has Been Created",
                                        "Error: Openning Invoice",
                                        0)  # this is making a popup box tell you what is wrong. This should never happend but just in case


root = Tk() #making root - Tk to open a window
master = Toplevel(root) # This opens thje master toplevel(root) for the pop up
master.destroy() #this destroys the master popup asoon as it is created. The reasoning behind this is because the
# master.destroy function needs something to reference if nothing is their so it does not cause a error that
#stops the program

center_window(510, 240) #runs said command with said functions
root.iconbitmap(default='just_logo.ico') #sets the icon of the windows

download_rss() #runs the said function

root.title('Offline Shopping') #sets the title of frame root
x = 'white' #changes x  to white
font = 5 #makes font 5
bgroot = 'white' #makes bgroot white

root.configure(background=bgroot) #changes the root configuration to said colour

var = IntVar() #runs for the buttons to get a variabl;e
variable = StringVar(root) #runs this so that the option menu gets variable
variable.set("1")  # default value

logoframe = Frame(root, bg=bgroot) #sets the logo frame
buttons_outer_frame = Frame(root, bg=bgroot) #sets the outer frmae
buttons_discontinued = Frame(buttons_outer_frame, bg=bgroot, highlightbackground="black", highlightcolor="black",
                             highlightthickness=1, width=100, height=100, bd=0) #sets the frame
buttons_specials = Frame(buttons_outer_frame, bg=bgroot, highlightbackground="black", highlightcolor="black",
                         highlightthickness=1, width=100, height=100, bd=0)# sets the frame
buttoncartinvoice = Frame(buttons_outer_frame, bg=bgroot) #sets this frame

clearnace = Label(buttons_outer_frame, text=('Clearnace'), fg='red', bg=bgroot) #makes the clreance label
recent_stock = Label(buttons_outer_frame, text=('Just In'), fg='red', bg=bgroot) #makes the just in label
add_to_cart_text = Label(buttoncartinvoice, text=('Item Number:'), fg='red', bg=bgroot) #makes the addtocat label

canvas = Canvas(logoframe, width=315, height=221, bg=bgroot) #makes a canvas for the logo
canvas.grid() #puts the canvas in the grid
img = PhotoImage(file="logo.gif") #sets the img to an said image
canvas.create_image(3, 1, anchor=NW, image=img) #places the image in canvas

# create the image button, image is above (top) the optional text
R1 = Radiobutton(buttons_discontinued, text="Clothing", justify=LEFT, variable=var, value=1, bg=bgroot,
                 command=sel) #makes the radio button
R2 = Radiobutton(buttons_discontinued, text="Fishing Gear", justify=LEFT, variable=var, value=2, bg=bgroot,
                 command=sel) #makes the radio button
R3 = Radiobutton(buttons_specials, text="Electronics", justify=LEFT, variable=var, value=3, bg=bgroot,
                 command=sel) #makes the radio button
R4 = Radiobutton(buttons_specials, text="Desktop PC", justify=LEFT, variable=var, value=4, bg=bgroot,
                 command=sel) #makes the radio button

add_to_cart = Button(buttoncartinvoice, text="Add to Cart", justify=LEFT, bg=bgroot, command=cart_item) #makes the add cart button
number = OptionMenu(buttoncartinvoice, variable, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10") #makes the option menu
number.config(bg=bgroot) #congfigures option menu
print_invoice = Button(buttoncartinvoice, text="Print Invoice", bg=bgroot, command=print_invoice) #makes the print invvoice button

invoice_open = Button(buttoncartinvoice, text="Open Invoice", justify=LEFT, bg=bgroot, command=open_invoice) #makes the invoice open button

logoframe.grid(row=0, column=0, sticky=W) #puts the said item in the said row/ colum/pading/sticky
buttons_outer_frame.grid(row=0, column=1, padx=(5, 0), sticky=W) #puts the said item in the said row/ colum/pading/sticky
clearnace.grid(row=0, column=0, sticky=S + W, padx=(5, 0), pady=(0, 48)) #puts the said item in the said row/ colum/pading/sticky
buttons_discontinued.grid(row=0, column=0, pady=(10, 5), sticky=N + E + W) #puts the said item in the said row/ colum/pading/sticky
R1.grid(row=0, column=0, sticky=W + S, pady=(5, 0)) #puts the said item in the said row/ colum/pading/sticky
R2.grid(row=0, column=1, sticky=E + S, padx=(9, 0)) #puts the said item in the said row/ colum/pading/sticky
recent_stock.grid(row=1, column=0, sticky=N + W, pady=(0, 40), padx=(5, 0)) #puts the said item in the said row/ colum/pading/sticky
buttons_specials.grid(row=1, column=0, pady=(10, 5), sticky=N + E + W) #puts the said item in the said row/ colum/pading/sticky
R3.grid(row=2, column=0, sticky=W + S, pady=(5, 0)) #puts the said item in the said row/ colum/pading/sticky
R4.grid(row=2, column=1, sticky=E + S) #puts the said item in the said row/ colum/pading/sticky
buttoncartinvoice.grid(row=2, column=0, pady=(0, 25), sticky=N) #puts the said item in the said row/ colum/pading/sticky
add_to_cart_text.grid(row=5, column=0, sticky=W) #puts the said item in the said row/ colum/pading/sticky
number.grid(row=5, column=1, sticky=W) #puts the said item in the said row/ colum/pading/sticky
add_to_cart.grid(row=6, column=0, sticky=E, padx=(0, 5)) #puts the said item in the said row/ colum/pading/sticky
print_invoice.grid(row=6, column=1, sticky=W) #puts the said item in the said row/ colum/pading/sticky
invoice_open.grid(row=7, column=0, pady=(5, 0), padx=(5, 0), ) #puts the said item in the said row/ colum/pading/sticky
root.mainloop()# makes the window run
