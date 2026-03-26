###################################################
# PLOTTING FUNCTIONS
# Will Schmid 
# Version 21 October 2024
###################################################
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator,FormatStrFormatter)

##########################################
# Explicitly set size of axes box in inches
##########################################
def set_size(w, h, ax=None):
    """
    Parameters
    ----------
    w: float
    h: float
    ax: matplotlib axis (optional)

    Returns
    -------
    nothing except the box of the most recent axis, or the axis "ax",
    sized at exactly w inches wide and h inches tall
    """

    # If no axis referenced, refer to most recent
    if not ax: 
        ax=plt.gca()

    # Left, right, top, bottom sizes
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # Set figure width and height in inches
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


#######################################
# Set up plot axes
#######################################
def plot_setup(xlabel, ylabel, 
               twin_axis=False, y2label=None,
               figsize = (5,4), auto_scale = False,
               axislinewidth=0.5, dpi = 300, scale_fonts = True,
               font = 'Arial', fontsize = 18, fontweight = 'light',
               xscale='linear',yscale='linear',y2scale='linear',
               xpad = 4.0, ypad = 4.0, y2pad = 4.0,
               xmin = None, xmax = None, xstep = None, xlim=None,
               ymin = None, ymax = None, ystep = None, ylim=None, 
               y2min = None, y2max = None, y2step = None, y2lim=None,
               xticks=None, yticks=None, y2ticks=None, 
               xfmt=None, yfmt=None, y2fmt=None, 
               minorticks = True, xminorticks = True, yminorticks = True,
               y2minorticks = True, nticks = 2, 
               tickwidth = 0.5, l_maj = 10, l_min = 6, 
               xclr = 'black', yclr = 'black', y2clr='black',
               title=None,
               ):
    """
    Minimum Parameters
    ------------------
    xlabel: str
    ylabel: str

    Important Optional Parameters
    -----------------------------
    twin_axis: bool (optional)
    y2label: str (optional)
    figsize: duple (optional)
    auto_scale: bool (optional)

    Returns
    -------
    Case 1: twin_axis = False (default):
        fig
            matplotlib figure containing new plotting axis "ax"
        ax
            matplotlib axis belonging to new parent figure "fig"
    Case 2: twin_axis = True:
        fig
            matplotlib figure containing twin plotting axes "ax" and "ax2"
        ax
            matplotlib axis belonging to new parent figure "fig", left axis
        ax2
            matplotlib axis belonging to new parent figure "fig", right axis

    Notes
    -----
    Examples:
    Single axis: ax = plot_setup('x','y')
    Twin axis: ax, ax2 = plot_setup('x','y1',twin_axis=True,'y2')

    Ticks:
    Major ticks can be formatted with xmax, xmin, and xstep (same for y and y2).
    By default, if xmax, xmin, and xstep are included, major ticks are drawn from
    xmin to xmax with spacing xstep, and the overall axis limits are xmin to xmax.
    If you desire to have the limits of the axis offset from the maximum and minimum,
    pass xlim as a separate argument. Tick lengths are set by l_maj (major) and l_min 
    (minor).
    Example: single axis with ticks at 0, 2, 4, 6, 8, and 10, with the overall x axis
    ranging from -0.5 to 10.5:
        ax = plot_setup('x','y',xmin=0,xmax=10,xstep=2,xlim=(-0.5,10.5))

    Coloring:
    To color the axis label and tick labels, use xclr, yclr, and/or y2clr
    Example: twin axes with red left axis and blue right axis labels and ticks:
        ax, ax2 = plot_setup('x','y1',twin_axis=True,'y2',yclr='red',y2clr='blue')
    (use colors from colorbrewer2.org, not default colors 'red','blue',etc. Many 
    colorbrewer colors can be loaded from "colors.py")

    Size:
    Set figsize duple, default (5,4), to give the size of the *axis box* 
    (not the entire image with labels) plot_setup calls the function set_size to ensure 
    the xy axis box is a specific size. Default units are inches: (5,4) gives a box that's 
    5 inches wide by 4 inches tall. All other defaults (font size, tick lengths, padding, 
    as well as cure line widths and marker sizes in "plot") are set to be aesthetically 
    pleasing, in Will's opinion, for a 5 inch x 4 inch figure. Font size will be true-to-size 
    in Illustrator and PowerPoint IF YOU DO NOT CHANGE THE SIZE AFTER IMPORTING THE FIGURE.

    Scaling and Aspect Ratio:
    Set auto_scale to True to use auto scaling. This is convenient when you are changing
    the size and aspect ratio from their defaults (5 inches wide by 4 inches tall). 
    Axis box line thickness, tick lengths and thicknesses, and fontsizes will all be scaled
    to maintain relatvie sizing and be visually similar to the default sizes.
    """

    # Font properties
    plt.rcParams['font.family'] = font
    plt.rcParams['font.weight'] = fontweight

    # Scaling for fig sizes other than (5,4)
    scaling = figsize[0]/5

    # Scale sizes and widths
    if auto_scale == True and scale_fonts == False:
        tickwidth = tickwidth*scaling
        axislinewidth = axislinewidth*scaling
        l_maj = l_maj*scaling
        l_min = l_min*scaling
        xpad = xpad*scaling
        ypad = ypad*scaling
        y2pad = y2pad*scaling
    elif auto_scale == True and scale_fonts == True:
        fontsize = fontsize*scaling
        tickwidth = tickwidth*scaling
        axislinewidth = axislinewidth*scaling
        l_maj = l_maj*scaling
        l_min = l_min*scaling
        xpad = xpad*scaling
        ypad = ypad*scaling
        y2pad = y2pad*scaling


    # Create axes
    fig, ax = plt.subplots(dpi=dpi)

    # x axis
    ax.set_xscale(xscale) # axis scale
    ax.set_xlabel(xlabel,  color=xclr, fontsize=fontsize, labelpad=xpad) # axis label, label color, font size, padding
    ax.tick_params(axis='x', labelcolor=xclr, labelsize=fontsize, pad=xpad) # tick color, tick label fontsize, padding
    if xstep != None:
        ax.set_xlim(xmin,xmax)   
        ax.set_xticks(np.arange(xmin,xmax+xstep,xstep))
    if xlim != None:
        ax.set_xlim(xlim)
    if xticks !=None:
        ax.set_xticks(xticks)
    if xfmt != None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt)) #example: '%.0f'
    if xscale == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=nticks))   
    
    # y (left) axis
    ax.set_yscale(yscale) # axis scale
    ax.set_ylabel(ylabel,  color=yclr, fontsize=fontsize, labelpad=ypad) # axis label, label color, font size, padding
    ax.tick_params(axis='y', labelcolor=yclr, labelsize=fontsize, pad=ypad) # tick color, tick label fontsize, padding
    if ystep != None:
        yrange = ymax-ymin
        ax.set_ylim(ymin-0.05*yrange,ymax+0.05*yrange)   
        ax.set_yticks(np.arange(ymin,ymax+ystep,ystep)) 
    if ylim != None:
        ax.set_ylim(ylim)
    if yticks != None:
        ax.set_yticks(yticks)
    if yfmt != None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt)) #example: '%.0f'
    if yscale == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=nticks))   

    # Global tick properties, left axis
    ax.tick_params(which='major',length=l_maj,direction='in',width=tickwidth)
    ax.tick_params(which='minor',length=l_min,direction='in',width=tickwidth)

    # Turn off minor ticks if requested
    if minorticks == False:
        ax.minorticks_off()

    # Turn on or off individual axis minor ticks
    ax.tick_params(axis='x', which='minor', bottom=xminorticks) 
    ax.tick_params(axis='y', which='minor', bottom=yminorticks) 

    # Width of axis lines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axislinewidth)

    # y2 (right) axis
    if twin_axis == True:
        ax2 = ax.twinx() # create twin axis 
        ax2.set_yscale(y2scale) # axis scale
        ax2.set_ylabel(y2label,  color=y2clr, fontsize=fontsize, labelpad=y2pad) # axis label, label color, font size, padding
        ax2.tick_params(axis='y', labelcolor=y2clr, labelsize=fontsize, pad=y2pad) # tick color, tick label fontsize, padding
        if y2step != None:
            y2range = y2max-y2min
            ax2.set_ylim(y2min-0.05*y2range,y2max+0.05*y2range)  
            ax2.set_yticks(np.arange(y2min,y2max+y2step,y2step)) 
        if y2lim != None:
            ax2.set_ylim(y2lim)
        if y2ticks !=None:
            ax2.set_yticks(y2ticks)
        if y2fmt != None:
            ax2.yaxis.set_major_formatter(FormatStrFormatter(y2fmt)) #example: '%.0f'
        if y2scale == 'linear':
            ax2.yaxis.set_minor_locator(AutoMinorLocator(n=nticks))   

        # Global tick properties, left axis
        ax2.tick_params(which='major',length=l_maj,direction='in',width=tickwidth)
        ax2.tick_params(which='minor',length=l_min,direction='in',width=tickwidth)

        # Turn off minor ticks if requested
        if minorticks == False:
            ax2.minorticks_off()

        # Turn on or off individual axis minor ticks
        ax2.tick_params(axis='y', which='minor', bottom=y2minorticks) 
        
        # Width of axis lines
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(axislinewidth)

    # Title
    ax.set_title(title,fontsize=fontsize)
        
    # Set size of axis box in inches
    set_size(figsize[0],figsize[1])
    
    # Return single axis or twin axes
    if twin_axis == True:
        return fig, ax, ax2
    else:
        return fig, ax


#########################################################
# Add secondary axis for extra row of labels (INCOMPLETE)
#########################################################
"""
NEEDS DOCUMENTATION
"""
def sec_axis(fig, ax, ticklabels, auto_scale=False, which='x',
            fontsize = 18, scale_fonts=True,
            tmin = None, tmax = None, tstep=None,
            axislinewidth=0.5, tickwidth=0.5,
            pad = 4.0, l_maj = 10,
            ):
        
    # Scaling
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scaling = bbox.width/5

    # Scale sizes and widths
    if auto_scale == True and scale_fonts == False:
        tickwidth = tickwidth*scaling
        axislinewidth = axislinewidth*scaling
        l_maj = l_maj*scaling
        pad = pad*scaling
    elif auto_scale == True and scale_fonts == True:
        fontsize = fontsize*scaling
        tickwidth = tickwidth*scaling
        axislinewidth = axislinewidth*scaling
        l_maj = l_maj*scaling
        pad = pad*scaling

    # Create secondary overlapping axis of choice
    if which == 'x':
        sec = ax.secondary_xaxis(location=0) # initialize axis
        if tstep !=None:
            sec.set_xticks(np.arange(tmin,tmax+tstep,tstep), labels=ticklabels, fontsize=fontsize) # tick properties
    elif which == 'y' or which == 'y2':
        sec = ax.secondary_yaxis(location=0) # initialize axis
        trange = tmax-tmin 
        if tstep !=None:
            sec.set_ylim(tmin-0.05*trange,tmax+0.05*trange) # axis range
            sec.set_yticks(np.arange(tmin,tmax+tstep,tstep), labels=ticklabels, fontsize=fontsize) # tick properties

    # Global tick properties
    sec.tick_params(which='major', length=l_maj, direction='in', pad=pad, width=tickwidth)

    # Axis linewidth
    for axis in ['top','bottom','left','right']:
        sec.spines[axis].set_linewidth(axislinewidth)

    return sec


#######################################
# Plot curve on axis ax
#######################################
def plot(fig, ax, xvar, yvar, 
        linestyle='-', color='black', label='_nolabel_',
        linewidth=2, markersize=10, auto_scale = False,
        markerfacecolor=None, markevery=None):
    """
    Minimum Parameters
    ------------------
    fig: matplotlib figure
    ax: matplotlib axis
    xvar: 1D array
    yvar: 1D array

    Returns
    -------
    nothing except a new curve on your axis "ax"

    Notes
    -----
    Procedure:
    1) Define axis with plot_setup(): "ax = plot_setup(xlabel,ylabel)"
    2) Plot curve with plot() by *passing the figure and axis just created as an argument*:
        "plot(fig, ax, xvar, yvar)"

    Legend and labels:
    To create a standalone legend with the next function legend(), use the label
    argument.

    Secondary axis:
    To plot on your secondary/twin/right axis, call that axis as the first argument.
    Example (plot on left and right axis, here defined as ax2):
        ax, ax2 = plot_setup('x','y1',twin_axis=True,'y2')
        plot(fig,ax,xvar,yvar)
        plot(fig,ax2,xvar,y2var)

    Autoscale:
    Set auto_scale = True if the parent axis was created with autoscaling - this will
    scale the linewidths and markersizes accordingly.
    """

    # Scaling
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scaling = bbox.width/5 

    if auto_scale == True:
        linewidth = linewidth*scaling
        markersize = markersize*scaling
    
    # Fill markers if no facecolor setting specified
    if markerfacecolor == None:
        markerfacecolor = color
    
    # Plot curve on axis
    ax.plot(xvar,yvar,linestyle,
                color=color,
                label=label,
                linewidth=linewidth,
                markersize=markersize,
                markerfacecolor = markerfacecolor,
                markevery=markevery,
                zorder=3)
    

################################################
# Plot vertical bar plot on axis ax (INCOMPLETE)
################################################
def bar(fig, ax, coords, heights,
        barcolors = ['black'],
        barwidth = 1, auto_scale = False, 
        labelcolors = False):
    """
    Minimum Parameters
    ------------------
    NEEDS DOCUMENTATION

    Returns
    -------
    nothing except a new barplot on your axis "ax"

    """

    # Scaling
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scaling = bbox.width/5 

    # Scale bar width and labels
    if auto_scale == True:
        barwidth = scaling*barwidth

    # Set to no color if none specified
    if barcolors == ['black']:
        for n in range(len(coords)-1):
            barcolors = barcolors.append('black')

    # Plot bar plot on axis
    ax.bar(coords,heights,width=barwidth,color=barcolors)

    # Set label colors if requested
    if labelcolors == True:
        [t.set_color(i) for (i,t) in zip(barcolors,ax.xaxis.get_ticklabels())]




#######################################
# Create stand-alone legend for axis ax
#######################################
def legend(fig, ax, 
           fancybox = False, dpi=300, figsize = (1,0.8), auto_scale = False, 
           font = 'Arial', fontsize = 12, fontweight = 'light',
           framealpha = 1, edgecolor = 'black',axislinewidth = 0.5, ncol=1):
    """
    Minimum Parameters
    ------------------
    fig: matplotlib figure
    ax: matplotlib axis

    Returns
    -------
    figl: matplotlib figure
        New stand-alone containing only the legend for input figure "fig"
    axl: matploblib axis
        Axis on returned figure "figl", containing only the legend for input figure "fig" 

    Notes
    -----
    - legend(fig,ax) uses labels and colors passed via each calling of plot(ax,...)
    - Example (creates two figures - the plot and the stand-alone legend):
        x = np.arange(-10,10,0.1)
        ax = plot_setup('x','y')
        plot(fig,ax,x,x**2,label=r'$y = x^2$',color=green)
        plot(fig,ax,x,x**3,label=r'$y = x^3$',color=purple)
        legend(fig,ax)
    """

    # Scaling
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scaling = bbox.width/5

    if auto_scale == True:
        fontsize = fontsize*scaling
        axislinewidth = axislinewidth*scaling  
    
    # Legend edge type (default straight corners)
    plt.rcParams["legend.fancybox"] = fancybox

    # Global font properties
    plt.rcParams['font.family'] = font
    plt.rcParams['font.weight'] = fontweight

    # Retrieve curves and labels from ax
    handles, labels = ax.get_legend_handles_labels()

    # Create legend axis
    figl, axl = plt.subplots(dpi=dpi)
    legend = axl.legend(handles, labels, loc='center',
               ncol=ncol, fontsize=fontsize,
               framealpha=framealpha,edgecolor=edgecolor)
    axl.xaxis.set_visible(False)
    axl.yaxis.set_visible(False)
    axl.set_axis_off()
    set_size(figsize[0],figsize[1])

    # Set line width of legend frame
    legend.get_frame().set_linewidth(axislinewidth)  
    return figl, axl


########################################################
# Plot countour map on axes ax (EXPERIMENTAL/INCOMPLETE)
########################################################
def contour(fig,ax,data,xind,yind,zind,
            auto_scale = False,
            zlabel=None,fontsize=18,
            cmap=plt.cm.Blues_r,
            levels=[],numlevels=7,
            zticks=[],tickformat=["{:2.2f}"],
            clabels = False,
            cbar_visible=True):
    """
    Minimum Parameters
    ------------------
    fig: matplotlib figure
    ax: matplotlib axis
    data: array
    xind: int
    yind: int
    zind: int

    Returns
    -------
    nothing except a new contour plot on your axis "ax"

    Notes
    -----
    This function currently follows slightly different conventions from plot().
    You must input your entire data matrix "data" without headers (i.e., "data"
    is the output of "np.genfromtxt()"). Instead of the variable arrays themselves,
    you must input the *column index* of those variables as xind, yind, and zind.
    This function produces a plot of data[:,zind] vs. data[:,xind] and data[:,yind]. 
    This function is still a work-in-progress and does not feature robust 
    auto-scaling.
    """
    
    # Scaling
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scaling = bbox.width/5

    if auto_scale == True:
        fontsize = 18*scaling
    
    # Get data columns for x and y vars
    xvar = data[:,xind]
    yvar = data[:,yind]

    # Meshgrid for plotting
    X,Y = np.meshgrid(np.unique(xvar),np.unique(yvar))

    # Initialize storage array for z variable
    Z = np.zeros((len(np.unique(xvar)),len(np.unique(yvar)))) 

    # Loop over xvar and yvar values and fill zvar array
    for m in range(len(np.unique(xvar))):
        
        data_m = data[data[:,xind]==np.unique(xvar)[m]]
        
        for n in range(len(np.unique(yvar))):
           
            Z[m,n] = data_m[n,zind] # height for current (x,y) pair

    # Set levels and ticks
    if levels == []:
        levels = np.linspace(np.amin(Z),np.amax(Z),numlevels)
    if zticks == []:
        zticks = levels

    # Create contour plot
    CS = ax.contourf(X, Y, Z.T, levels=levels, cmap=cmap)
    cbar = plt.colorbar(CS)
    if zlabel != None:
        cbar.set_label(label=zlabel,size=fontsize)
    cbar.set_ticks(zticks)
    cbar.set_ticklabels([tickformat.format(i) for i in cbar.get_ticks()],size=fontsize) 

    # Colorbar labels
    if clabels == True:
        ax.clabel(CS, colors = 'black',rightside_up=1, inline=0, fontsize=fontsize-4)

    # Remove colorbar?
    if cbar_visible == False:
        cbar.remove()



####################################################
# Find index of array element closest to some value
####################################################
def find_nearest(array, value):
    """
    Returns index of value in NP array "array" closest to desired float "value".
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
