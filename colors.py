###################################################
# PLOTTING COLORS
# Will Schmid 
# Version 18 October 2024
###################################################
# Library
import matplotlib.pyplot as plt

# Single colors
blue = "#08519C"; black = "#000000"; green = "#1B7837"; red = "#A50F15"; orange = '#f16913';
purple = "#6e016b"; copper = "#b87333"; water = "d4f1f9"; off_white = "#f7f7f7"

# Diverging colors
dark_red = "#ca0020"; light_red = "#f4a582"; light_blue = "#92c5de"; dark_blue = "#0571b0"
dark_purple = "#7b3294"; light_purple = "#c2a5cf"; light_green = "#a6dba0"; dark_green = "#008837"
dark_orange = "#e66101"; light_orange = "#fdb863"; light_lavender = "#b2abd2"; dark_lavender = "#5e3c99"

# Six shade color schemes
blues = ['#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
oranges = ['#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#8c2d04']
reds = ['#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']
purples = ['#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#4a1486']
greens = ['#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#005a32']
greys = ['#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']

# Colormaps for contour plots
Greys = plt.cm.Greys_r
Hot = plt.cm.hot
Blues = plt.cm.Blues_r
Oranges = plt.cm.Oranges_r
Greens = plt.cm.Greens_r
Reds = plt.cm.Reds_r
Purples = plt.cm.Purples_r
YellowRed = plt.cm.YlOrRd_r
GreenBlue = plt.cm.YlGnBu_r