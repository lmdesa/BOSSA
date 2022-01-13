''' 
# online material accompanying the papers: 
    Chruslinska, M. & Nelemans, G., 2019, MNRAS, 488, 5300
    Chruslinska, M., Jerabkova, T., Nelemans, G., Yan, Z.,2020, A&A
# this python script allows to visualize the data (plot the SFRD(Z,z) distribution and CDF at a given redshift)
# usage: python plot_SFRD_Z_z.py
# adjust the data input file name (input_file); see example use at the bottom
# requires 'Time_redshift_deltaT.dat' and input file with the data for a chosen variation (any of the '*FOH_z_dM.dat')
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.colors as colors

labelsize=21
ticklabsize=17

def solar_metallicity_scales():
    Asplund09=[0.0134,8.69]
    AndersGrevesse89=[0.017,8.83]
    GrevesseSauval98=[0.0201,8.93]
    Villante14=[0.019,8.85]
    scale_ref=np.array(['Asplund09','AndersGrevesse89','GrevesseSauval98','Villante14'])
    Z_FOH_solar=np.array([Asplund09,AndersGrevesse89,GrevesseSauval98,Villante14])
    return scale_ref, Z_FOH_solar

def FOH2ZZ(foh,solar_Z_scale='AndersGrevesse89'):
    '''convert from 12+log[O/H] to ZZ'''
    scale_ref, Z_FOH_solar=solar_metallicity_scales()
    idx=np.where(scale_ref==solar_Z_scale)[0][0]
    Zsun,FOHsun = Z_FOH_solar[idx]    
    logZ = np.log10(Zsun) + foh - FOHsun
    ZZ=10**logZ
    return ZZ

def ZZ2FOH(zz,solar_Z_scale='AndersGrevesse89'):
    '''convert from ZZ to 12+log[O/H] '''
    scale_ref, Z_FOH_solar=solar_metallicity_scales()
    idx=np.where(scale_ref==solar_Z_scale)[0][0]
    Zsun,FOHsun = Z_FOH_solar[idx]
    foh = np.log10(zz)-np.log10(Zsun)+FOHsun
    return foh

def smooth(d, c=2):
    if c == 0:
        return d
    x = np.zeros(len(d))
    x[0] = (d[1]+d[0])/2
    x[-1] = (d[-1]+d[-2])/2
    x[1:-1] = (2*d[1:-1]+d[:-2]+d[2:])/4
    return smooth(x, c=c-1)

#(array) oxygen to hydrogen abundance ratio ( FOH == 12 + log(O/H) )
# as used in the calculations - do not change
FOH_min, FOH_max = 5.3, 9.7
FOH_arr = np.linspace( FOH_min,FOH_max, 200)
dFOH=FOH_arr[1]-FOH_arr[0]

def get_plot_data(input_file,zmin=0.,zmax=4):

    #read time, redshift and timestep as used in the calculations
    #starts at the highest redshift (z=z_start=10) and goes to z=0
    time, redshift_global, delt = np.loadtxt('Time_redshift_deltaT.dat',unpack=True) 
    #reading mass per unit (comoving) volume formed in each z (row) - FOH (column) bin
    data=np.loadtxt(input_file)
    image_data=np.array( [data[ii]/(1e6*delt[ii]) for ii in range(len(delt))] )#fill the array with SFRD(FOH,z)

    redshift=redshift_global
    #select the interesting redshift range
    if( zmax!=10 or zmin!=0 ):
        idx= np.where(np.abs(np.array(redshift)-zmax)==np.abs(np.array(redshift)-zmax).min())[0][0] 
        idx0= np.where(np.abs(np.array(redshift)-zmin)==np.abs(np.array(redshift)-zmin).min())[0][0] 
        image_data = image_data[idx:idx0]
        redshift=redshift_global[idx:idx0]
        delt=delt[idx:idx0]

    image_data/=dFOH
    return image_data, redshift, delt

def plot_data_single_panel(input_file, zmin=0.,zmax=10,solar_Z_scale='AndersGrevesse89'):

    #read data
    image,redshift,delt = get_plot_data(input_file,zmin,zmax)

    scale_ref, Z_FOH_solar=solar_metallicity_scales()
    idx=np.where(scale_ref==solar_Z_scale)[0][0]
    Zsun,FOHsun = Z_FOH_solar[idx]
    FOHymin,FOHymax=5.39,10.
  
    fig1=plt.figure(figsize=(8.5,6))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=3)
    ax11 = ax1.twinx()

    #plotting
    vmin=3e-5
    vmax=0.22
    idx_peak_SFRH = np.where( np.abs(np.array(redshift)-1.8)==np.min(np.abs(np.array(redshift)-1.8)) )[0][0]
    c='saddlebrown'
    image1 = ndimage.gaussian_filter(image,1,0)
    pcm=ax1.pcolormesh(redshift, FOH_arr, np.transpose(image1),norm=colors.LogNorm(vmin=vmin, vmax=vmax ))#,cmap='OrRd' )
    levels=[0.01,0.03,0.05,0.1]
    CS=ax1.contour(redshift,FOH_arr, np.transpose(image1),levels=levels,colors=('brown',),linestyles=('-',),linewidths=(1.5,))
    ax1.clabel(CS, fmt = '%g', colors ='k', fontsize=15) #contour line labels
    SFRD_atz_Z = np.array([image1[:][ii] for ii in range(len(image1[:,0]))])
    maxima = np.array([SFRDi[2:].max() for SFRDi in SFRD_atz_Z ])
    indices=[np.where( np.abs(maxi-SFRDi[2:])==np.min(np.abs(maxi-SFRDi[2:])))[0][-1] for maxi,SFRDi in zip(maxima,SFRD_atz_Z)]
    FOHmax=np.array(FOH_arr[indices])+(FOH_arr[1]-FOH_arr[0])*0.5
    ax1.plot(redshift,smooth(np.array(FOHmax),10),lw=2.5,c=c,label='peak metallicity')

    #prepare labels for y-axes
    fig1.canvas.draw()
    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels2=[]
    for l in labels[:]:
            #print l
            if l=='': labels2.append('')
            else: 
                 labels2.append( str('%.1g'%(FOH2ZZ(float(l),solar_Z_scale))) )
    labels1=[]
    labels1.append('')
    labels1+=labels[1:]
    ax1.set_yticklabels(labels1)
    ax11.set_yticklabels(labels2)

    #add legend
    lgnd = ax1.legend( loc='upper right', prop={'size':int(14)},\
            fancybox=True,numpoints=1,scatterpoints = 1)#,frameon=False)
    frame = lgnd.get_frame()
    frame.set_alpha(0.85)

    #adjust size
    w,h=fig1.get_size_inches()
    fig1.set_size_inches(w,h+2.2, forward=True)
    ttop,lleft,rright,bbottom=0.88,0.12,0.85,0.08
    plt.subplots_adjust(left=lleft, bottom=bbottom, right=rright, top=ttop, wspace=0.01, hspace=0.0)

    #add colorbar
    cbaxes = fig1.add_axes([lleft, ttop, rright-lleft, 0.01])
    cb=plt.colorbar(pcm, orientation='horizontal', cax=cbaxes)
    cb.set_label(label=r'$\rm \frac{SFRD}{\Delta z \Delta Z_{O/H}} [M_{\odot}/Mpc^{3}yr]$',\
                size=labelsize+2, labelpad=15)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=ticklabsize)

    #add description    
    ax1.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False, #top ticks off
            labelbottom=True,
            labeltop=False,
            labelleft=True,
            labelright=False,
            labelsize=ticklabsize) # labels along the bottom edge are off
    ax11.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=True,
            labelsize=ticklabsize) # labels along the bottom edge are off

    ax1.set_xlabel('redshift')
    ax1.set_ylim([FOHymin,FOHymax])
    ax11.set_ylim([FOHymin,FOHymax])
    ax11.set_ylabel('metallicity - Z',fontsize=labelsize )
    ax1.set_ylabel('12 + log(O/H)',fontsize=labelsize )
    ax1.set_xlim([min(redshift), max(redshift)])
    ax1.set_xlabel('z --- redshift', fontsize=labelsize)

    cann,cann2='gray','gray'
    ax1.plot( redshift, [FOHsun for z in redshift], ls='--', c=cann2,lw=1.5)
    ax1.annotate(r'Z$_{\odot}$', xy=(redshift[5],ZZ2FOH(Zsun,solar_Z_scale)+0.0005),\
     xycoords='data',fontsize=labelsize, fontweight='bold',color='k')
    ax1.plot( redshift, [ZZ2FOH(0.1*Zsun,solar_Z_scale) for z in redshift], ls='--', c=cann2,lw=1.5)
    ax1.annotate(r'0.1 Z$_{\odot}$', xy=(redshift[8],ZZ2FOH(0.1*Zsun,solar_Z_scale)+0.0005),\
     xycoords='data',fontsize=labelsize, fontweight='bold',color=cann)
    cann,cann2='silver','gray'
    ax1.plot( redshift, [ZZ2FOH(0.01*Zsun,solar_Z_scale) for z in redshift], ls='--', c=cann2,lw=1.5)
    ax1.annotate(r'0.01 Z$_{\odot}$', xy=(redshift[9],ZZ2FOH(0.01*Zsun,solar_Z_scale)+0.00005),\
     xycoords='data',fontsize=labelsize, fontweight='bold',color=cann)

    plt.show()

def prepare_CDF(SFRD_data):

    #NORMALIZE the input data
    mtot_z = [np.sum(SFRD_data[:][ii]) for ii in range(SFRD_data.shape[0])]
    SFRD_normed = np.array([ [(SFRD_data[ii][j])/mtot_z[ii] for\
                     j in range(SFRD_data.shape[1])] for ii in range(SFRD_data.shape[0])]) 
    #CALCULATE the cumulative sum of the data
    Z_cumsum = np.array([ [np.sum(SFRD_normed[ii][:j]) for j in range(SFRD_data.shape[1])]\
                             for ii in range(SFRD_data.shape[0])])
    Z_cumsum=np.transpose(Z_cumsum)
    return Z_cumsum

def plot_CDF(input_file, zzi=[0], color='g',label='' ):

    image,redshift,delt = get_plot_data(input_file,zmin=0,zmax=10)
    Z_cumsum=prepare_CDF(image)
    redshift=np.array(redshift)

    fig1=plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.tick_params(axis='x', which='major', labelsize=ticklabsize)
    ax1.tick_params(axis='y', which='major', labelsize=ticklabsize)
    ax1.set_ylabel('fraction of SFRD at Z$_{O/H}<x$',fontsize=labelsize)
    ax1.tick_params(axis='x', which='major', labelsize=ticklabsize)
    ax1.tick_params(axis='y', which='major', labelsize=ticklabsize)
    ax1.set_ylim([0.,1.05])
    ax1.set_xlim([6.,10])
    ax1.set_xlabel('12+log(O/H)',fontsize=labelsize)

    lw=2
    alpha=1
    colors = plt.cm.jet(np.linspace(0,1,len(zzi)))
    for z,c in zip(zzi,colors):
        iz=np.where( np.abs(redshift-z)==np.min( np.abs(redshift-z) ) )[0][0]
        frac_Z=Z_cumsum[:,iz] #CDF at z=zzi
        if(len(zzi)==1):
                ax1.plot(FOH_arr, frac_Z,c=color,ls='-',lw=lw,label=label+'; z='+str(z),alpha=alpha)
        else:
                ax1.plot(FOH_arr, frac_Z,c=c,ls='-',lw=lw,label='z='+str(z),alpha=alpha)
    return Z_cumsum

''' Example use: '''
''' plot the SFRD(Z,z) distribution '''
# solar metallicity is shown in the figure to guide the eye; the conversion can be changed
# (see function solar_metallicity_scales() and keyword solar_Z_scale below)
input_file='302w14vIMF3aeh_FOH_z_dM.dat'
plot_data_single_panel(input_file, zmin=0.,zmax=10,solar_Z_scale='AndersGrevesse89')

''' example CDF plot comparing the high-Z extreme for different IMF assumptions at redshift=zzi '''
zzi=[0]
input_file='302w14vIMF3aeh_FOH_z_dM.dat'
image,redshift,delt = get_plot_data(input_file,zmin=0,zmax=10)
Z_cumsum = plot_CDF(input_file, zzi=zzi ,color='r',label='IGIMF3' )

input_file='302w14vIMF2aeh_FOH_z_dM.dat'
image,redshift,delt = get_plot_data(input_file,zmin=0,zmax=10)
Z_cumsum = plot_CDF(input_file, zzi=zzi, label='IGIMF2' )

input_file='302w14_FOH_z_dM.dat'
image,redshift,delt = get_plot_data(input_file,zmin=0,zmax=10)
Z_cumsum = plot_CDF(input_file, zzi=zzi, color='b',label='universal IMF' )

plt.legend(loc='best',ncol=1)
plt.tight_layout()
plt.show()
