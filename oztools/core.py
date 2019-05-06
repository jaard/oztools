import numpy as np
#from numba import jit
import scipy.io as sio
import pandas as pd
import datetime as dt
import seawater as sw
import matplotlib.pyplot as plt
import gsw
import time, sys

def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    sys.stdout.write("\r" + str(text))
    sys.stdout.flush()
    
def fwd_matdiv(B, A):
    soln = np.linalg.lstsq(A.T, B.T)[0].T
    return soln

def gauss(x, *args):

    '''
    function y = gauss(x, s, m)

    Computes the Gaussian function of the input array

    y=exp(-(x-m).^2./s.^2)./(sqrt(2*pi).*s)
    Bronstein p. 81
    '''

    try:
        m = args[1]
    except IndexError:
        m = 0
    try:
        s = args[0]
    except IndexError:
        s = 1

    y = np.exp(-(x - m)**2 / s**2) / (np.sqrt(2 * np.pi) * s)

    return y


# @jit  # <-- tells the numba library to optimize this function
def obana(x, posx, posy, gridx, gridy, r1x, r2x, *args, **kwargs):

    '''
    function new_z=obana2(z, x,    y,    new_x, new_y, r_x_inf, r_x_cut, r_y_inf, r_y_cut)
                          x, posx, posy, gridx, gridy, r1x,     r2x,     r1y,     r2y

    averaging with gaussian weights

    Parameters
    ----------
    z		- old values
    x		- old x-positions
    y		- old y-positions
    new_x		- new x-positions
    new_y		- new y-positions
    r_x_inf		- influence radius in x-direction
    r_x_cut		- cut-off radius in x-direction
    [r_y_inf]	- influence radius in y-direction
    [r_y_cut]	- cut-off radius in y-direction
    fillnans    - bool, sets if points containing nans are filled

    output :	new_z		- new values

    influence and cut-off radii may be given as vectors or matrices
    defining values for each grid point of the output array
    '''

    if kwargs:
        fillnans = kwargs.get('fillnans', False)
        # print(kwargs)
    else:
        fillnans = False

    # make isotropic if no y values are specified  <-- removed try/catch since it's not supported by numba
    if len(args) > 1:
        r2y = args[1]
    else:
        r2y = r2x
    if args:
        r1y = args[0]
    else:
        r1y = r1x

    # try:
    #     r2y = args[1]
    # except IndexError:
    #     r2y = r2x
    # try:
    #     r1y = args[1]
    # except IndexError:
    #     r1y = r1x

    # find axis along which radii (if vectors and not matrices) are given
    radshp = np.shape(r1x)
    grdshp = np.shape(gridx)

    # blow up influence and cut-off radii
    if np.shape(r1x) == ():
        r1x = np.tile(r1x, grdshp)
    elif np.size(np.shape(r1x)) == 1:
        if radshp[0] == grdshp[0]:
            r1x = np.tile(reshape(r1x, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r1x = np.tile(reshape(r1x, (radshp[0], 1), grdshp[0]))

    if np.shape(r2x) == ():
        r2x = np.tile(r2x, grdshp)
    elif np.size(np.shape(r2x)) == 1:
        if radshp[0] == grdshp[0]:
            r2x = np.tile(reshape(r2x, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r2x = np.tile(reshape(r2x, (radshp[0], 1), grdshp[0]))

    if np.shape(r1y) == ():
        r1y = np.tile(r1y, grdshp)
    elif np.size(np.shape(r1y)) == 1:
        if radshp[0] == grdshp[0]:
            r1y = np.tile(reshape(r1y, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r1y = np.tile(reshape(r1y, (radshp[0], 1), grdshp[0]))

    if np.shape(r2y) == ():
        r2y = np.tile(r2y, grdshp)
    elif np.size(np.shape(r2y)) == 1:
        if radshp[0] == grdshp[0]:
            r2y = np.tile(reshape(r2y, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r2y = np.tile(reshape(r2y, (radshp[0], 1), grdshp[0]))

    r1x = r1x.ravel()
    r2x = r2x.ravel()
    r1y = r1y.ravel()
    r2y = r2y.ravel()

    # setup input and target vector positions

    si = np.shape(gridx)
    posx = posx.ravel()
    posy = posy.ravel()
    gridx = gridx.ravel()
    gridy = gridy.ravel()
    x = x.ravel()
    m = np.shape(gridx)[0]

    # reset output
    y = np.nan * np.zeros(m)

    # loop over each output value
    pc = 0
    # print('\nOBANA: ')

    for ii in np.arange(m):
        # display process
        update_progress(ii/m)
        #if (ii / (m-2) > pc):
        #    print('\rOBANA: ' + str(int(pc * 100)) + ' %')
        #    pc = np.round(pc + 0.1, decimals=1)

        # positions difference
        dx = gridx[ii] - posx
        dy = gridy[ii] - posy

        # norm with cutoff radius
        jj = np.sqrt((dx/r2x[ii]) ** 2 + (dy/r2y[ii]) ** 2)

        # select only values within cutoff radius
        jj = np.where((abs(jj) < 1) * np.invert(np.isnan(jj)))
        if np.size(jj) != 0:

            # norm with inflence radius
            d = np.sqrt((dx[jj]/r1x[ii]) ** 2 + (dy[jj]/r1y[ii]) ** 2)

            # get factors using a gauss distribution
            d[:] = gauss(d, 1)

            # sum up values
            s = np.nansum(d, 0)

            if s > 0:

                if fillnans:
                    y[ii] = np.nansum(d*x[jj]) / s
                    # <-- nans are filled if there is data within radius
                else:
                    y[ii] = np.dot(d, x[jj]) / s
                    # <-- returns 'nan' if nans are present within radius

    update_progress(1)
    # reshape to GridSize
    y = np.reshape(y, si)

    return y


def obana_vector(x, posx, posy, gridx, gridy, r1x, r2x, *args):

    '''
    function new_z=obana2(z, x,    y,    new_x, new_y, r_x_inf, r_x_cut, r_y_inf, r_y_cut)
                          x, posx, posy, gridx, gridy, r1x,     r2x,     r1y,     r2y

    averaging with gaussian weights

    Parameters
    ----------
    z		- old values
    x		- old x-positions
    y		- old y-positions
    new_x		- new x-positions
    new_y		- new y-positions
    r_x_inf		- influence radius in x-direction
    r_x_cut		- cut-off radius in x-direction
    [r_y_inf]	- influence radius in y-direction
    [r_y_cut]	- cut-off radius in y-direction

    output :	new_z		- new values

    influence and cut-off radii may be given as vectors or matrices
    defining values for each grid point of the output array
    '''

    # make isotropic if no y values are specified
    try:
        r2y = args[1]
    except IndexError:
        r2y = r2x
    try:
        r1y = args[1]
    except IndexError:
        r1y = r1x

    # find axis along which radii (if vectors and not matrices) are given
    radshp = np.shape(r1x)
    grdshp = np.shape(gridx)

    # blow up influence and cut-off radii
    if np.shape(r1x) == ():
        r1x = np.tile(r1x, grdshp)
    elif np.size(np.shape(r1x)) == 1:
        if radshp[0] == grdshp[0]:
            r1x = np.tile(reshape(r1x, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r1x = np.tile(reshape(r1x, (radshp[0], 1), grdshp[0]))

    if np.shape(r2x) == ():
        r2x = np.tile(r2x, grdshp)
    elif np.size(np.shape(r2x)) == 1:
        if radshp[0] == grdshp[0]:
            r2x = np.tile(reshape(r2x, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r2x = np.tile(reshape(r2x, (radshp[0], 1), grdshp[0]))

    if np.shape(r1y) == ():
        r1y = np.tile(r1y, grdshp)
    elif np.size(np.shape(r1y)) == 1:
        if radshp[0] == grdshp[0]:
            r1y = np.tile(reshape(r1y, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r1y = np.tile(reshape(r1y, (radshp[0], 1), grdshp[0]))

    if np.shape(r2y) == ():
        r2y = np.tile(r2y, grdshp)
    elif np.size(np.shape(r2y)) == 1:
        if radshp[0] == grdshp[0]:
            r2y = np.tile(reshape(r2y, (radshp[0], 1), grdshp[1])).T
        elif radshp[0] == grdshp[1]:
            r2y = np.tile(reshape(r2y, (radshp[0], 1), grdshp[0]))

    r1x = r1x.flatten()
    r2x = r2x.flatten()
    r1y = r1y.flatten()
    r2y = r2y.flatten()

    # setup input and target vector positions

    si = np.shape(gridx)
    posx = posx.flatten()
    posy = posy.flatten()
    gridx = gridx.flatten()
    gridy = gridy.flatten()
    x = x.flatten()
    m = np.shape(gridx)[0]
    n = np.shape(posx)[0]

    # reset output
    y = np.nan * np.zeros(m)

    ram = 200
    pointMegs = n * 4 / 1000000
    megs = m * pointMegs
    chunkPoints = int(np.floor(ram / pointMegs))

    if chunkPoints < 1:
        print('Memory insufficient! Silently aborting...')
        return

    chunkMegs = chunkPoints * pointMegs
    chunks = int(np.ceil(m / chunkPoints))
    print('Chunk size is ' + str(chunkMegs) + ' megabytes.')
    print('Number of chunks is ' + str(chunks))

    # loop and vectorize
    for chunk in np.arange(0, chunks):

        print('chunk '+str(chunk+1))

        start = chunk * chunkPoints
        end = (chunk + 1) * chunkPoints
        if end > m-1:
            end = None

        gridx_chunk, posx_chunk = np.meshgrid(gridx[start:end], posx)
        gridy_chunk, posy_chunk = np.meshgrid(gridy[start:end], posy)

        # positions difference
        dx_chunk = gridx_chunk - posx_chunk
        dy_chunk = gridy_chunk - posy_chunk

        # r2x_chunk = np.tile(r2x[start:end], (n, 1))  # cutoff radii
        # r2y_chunk = np.tile(r2y[start:end], (n, 1))
        r1x_chunk = np.tile(r1x[start:end], (n, 1))  # influence radii
        r1y_chunk = np.tile(r1y[start:end], (n, 1))
        x_chunk = np.tile(x, (chunkPoints, 1)).T  # data


        # # norm with cutoff radius
        # jj = np.sqrt((dx / r2x) ** 2 + (dy / r2y) ** 2)
        #
        # # select only values within cutoff radius
        # ii, jj = np.where(abs(jj) < 1)
        # if np.size(jj) != 0:

        # norm with inflence radius
        d = (np.sqrt((dx_chunk / r1x_chunk) ** 2 + (dy_chunk / r1y_chunk) ** 2))

        # get factors using a gauss distribution
        d = gauss(d, 1)

        # sum up values
        s = np.nansum(d, 0)

        y[start:end] = np.einsum('ij,ij->i', d.T, x_chunk.T) / s

    # reshape to GridSize
    y = np.reshape(y, si)

    return y


def psdist(lat1, lon1, lat2, lon2, **kwargs):

    '''
    Calculates the distance according to the "plane sailing method".
    Same as in the seawater library, but vectorized.
    The unit of the output is kilometers.
    '''

    DEG2RAD = (2 * np.pi / 360)
    RAD2DEG = 1 / DEG2RAD
    DEG2MIN = 60
    DEG2NM = 60
    NM2KM = 1.8520

    if type(lat1) != np.ndarray:
        lat1 = np.array(lat1)
        lon1 = np.array(lon1)
        lat2 = np.array(lat2)
        lon2 = np.array(lon2)

    # # store inut shape and flatten for calculations
    # sp = np.shape(lat1)
    # lat1 = lat1.flatten()
    # lon1 = lon1.flatten()
    # lat2 = lat2.flatten()
    # lon2 = lon2.flatten()

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    latrad1 = abs(lat1 * DEG2RAD)
    latrad2 = abs(lat2 * DEG2RAD)
    dep = np.cos((latrad2 + latrad1) / 2) * dlon
    dist = DEG2NM * NM2KM * (dlat ** 2 + dep ** 2) ** 0.5  # in km

    # # restore to input shape
    # dist = np.reshape(dist, sp)

    # calculate angle to X axis
    angle = np.angle(dep + dlat * 1j, 1)  # 1: degrees 2: radians

    return dist, angle


def dist2coast(points_lat, points_lon):

    '''
    Calculates the distance to the nearest coastal point, based on the
    coastline file that is included in MATLAB.
    Will try to do this with GSHHS shapefiles in the future.
    '''
    coast = sio.loadmat('/Users/jhauschildt/Documents/Python/coastlines/coast.mat')
    c_lons = coast['long']
    c_lats = coast['lat']

    if np.size(points_lat) == 1:
        p_lons = np.tile(points_lon, c_lons.shape)
        p_lats = np.tile(points_lat, c_lats.shape)
        dists, angles = psdist(p_lats, p_lons, c_lats, c_lons)
        minind = np.nanargmin(dists)
        coastdist = dists[minind]
        coastangle = angles[minind]
    else:
        coastdist = np.empty_like(points_lat, dtype='float')
        coastangle = np.empty_like(points_lat, dtype='float')
        sp = np.shape(coastdist)
        coastdist = coastdist.ravel()
        coastangle = coastangle.ravel()
        points_lon = points_lon.ravel()
        points_lat = points_lat.ravel()
        for ll in range(len(coastdist)):
            p_lons = np.tile(points_lon[ll], c_lons.shape)
            p_lats = np.tile(points_lat[ll], c_lats.shape)
            dists, angles = psdist(p_lats, p_lons, c_lats, c_lons)
            minind = np.nanargmin(dists)
            coastdist[ll] = dists[minind]
            coastangle[ll] = angles[minind]
        coastdist = np.reshape(coastdist, sp)
        coastangle = np.reshape(coastangle, sp)

    return coastdist, coastangle


def uvrot(u, v, ang):

    '''
    rotates coordinate system (positive: counter-clockwise)
    takes angle in degrees

    Parameters
    ----------
    u       - zonal veocity
    v       - meridional velocity
    ang     - angle (in degrees) to rotate

    Returns
    -------
    ru   - rotated u
    rv   - rotated v
    '''

    ang = ang * np.pi / 180
    ru = u * np.cos(ang) + v * np.sin(ang)
    rv = v * np.cos(ang) - u * np.sin(ang)

    return ru, rv


def read_bottlefile(fname):

    '''
    Reads a ctd bottle file and returns a pandas dataframe
    '''

    # get names from header
    lineind = -1
    with open(fname) as f:
        for line in f:
            lineind += 1
            if line.startswith(' '):
                break
            else:
                headerline = line
    for ind in enumerate(headerline):
        names = headerline[ind[0]:]
        if names.startswith('prf'):
            # names = names[18:]
            break
    names = names.split(sep=':')
    btl_dataframe = pd.read_csv(fname, skiprows=lineind,
                                delim_whitespace=True, names=names)
    return btl_dataframe


def read_pangaea(fname):

    '''
    Reads a .tsv CTD file downloaded from PANGAEA
    and returns a pandas dataframe
    '''

    # find end of the header
    lineind = 0
    with open(fname, encoding='utf-8') as f:
        for line in f:
            lineind += 1
            if line.startswith('*/'):
                break

    pga_dataframe = pd.read_csv(fname, sep='\t', header=lineind,
                                encoding='utf-8')

    return pga_dataframe


def min2decdeg(lon_or_lat):

    '''
    Converts DEGREES.MINUTES to DEGREES.DECIMALDEGREES
    '''
    lon_or_lat = np.modf(lon_or_lat)[1] + np.modf(lon_or_lat)[0] / 0.6
    return lon_or_lat


def jday2date(year, jday):
    jday_dt = dt.datetime(year, 1, 1) + dt.timedelta(jday-1)
    return jday_dt


def date2jday(jday_dt):
    jday = jday_dt.timetuple().tm_yday
    year = jday_dt.timetuple().tm_year
    return year, jday


def datenum2datetime(datenum):
    #  + dt.timedelta(days=float(x) % 1)
    datetime = [dt.datetime.fromordinal(int(x)) - dt.timedelta(days=366)
                + dt.timedelta(days=float(x) % 1) for x in datenum]
    # datetime = [x + dt.timedelta(minutes=round(x.second/60)) for x in datetime]
    # for ind, x in enumerate(datetime):
    #     x.replace(minute=x.minute+round(x.second/60))
    #     x.replace(second=0)
    #     x.replace(microsecond=0)
    #     datetime[ind] = x
    return datetime



def distmatrix(lonmesh, latmesh):

    # Calculates the distances in x and y direction of the input array
    # The output is one element less than the input in each direction

    DEG2RAD = (2 * np.pi / 360)
    RAD2DEG = 1 / DEG2RAD
    DEG2MIN = 60
    DEG2NM = 60
    NM2KM = 1.8520

    # distances in x-direction
    lon3d = np.stack((lonmesh[1:-1, 0:-2], lonmesh[1:-1, 1:-1]), axis=2)
    lat3d = np.stack((latmesh[1:-1, 0:-2], latmesh[1:-1, 1:-1]), axis=2)
    dlon = np.diff(lon3d)[:, :, 0]
    latrad = abs(lat3d * DEG2RAD)
    dep = np.cos((latrad[:, :, 1] + latrad[:, :, 0]) / 2) * dlon
    dlat = np.diff(lat3d)[:, :, 0]
    x_distance = DEG2NM * NM2KM * np.sqrt(dlat**2 + dep**2)  # in km

    # distances in y-direction
    lon3d = np.stack((lonmesh[0:-2, 1:-1], lonmesh[1:-1, 1:-1]), axis=2)
    lat3d = np.stack((latmesh[0:-2, 1:-1], latmesh[1:-1, 1:-1]), axis=2)
    dlon = np.diff(lon3d)[:, :, 0]
    latrad = abs(lat3d * DEG2RAD)
    dep = np.cos((latrad[:, :, 1] + latrad[:, :, 0]) / 2) * dlon
    dlat = np.diff(lat3d)[:, :, 0]
    y_distance = DEG2NM * NM2KM * np.sqrt(dlat**2 + dep**2)  # in km

    distance = np.zeros_like(lon3d)
    distance[:, :, 0] = x_distance
    distance[:, :, 1] = y_distance
    return distance


def sstGrads(lon, lat, sst):

    # Calculates SST gradient
    longrid = np.meshgrid(lon, lat)[0]
    latgrid = np.meshgrid(lon, lat)[1]

    distgrid = distmatrix(longrid, latgrid)
    x_dist = distgrid[:, :, 0]  # distances btw. grid points in x direction
    y_dist = distgrid[:, :, 1]  # distances btw. grid points in y direction

    # SST difference in x-direction
    sst3d_xshift = np.ma.dstack((sst[1:-1, 0:-2], sst[1:-1, 1:-1]))
    sst3d_xshift[sst3d_xshift == -32767] = 'nan'
    dSST_dx = np.ma.diff(sst3d_xshift)[:, :, 0] / x_dist  # gradient in °C/km
    # SST difference in y-direction
    sst3d_yshift = np.ma.dstack((sst[0:-2, 1:-1], sst[1:-1, 1:-1]))
    sst3d_yshift[sst3d_yshift == -32767] = 'nan'
    dSST_dy = np.ma.diff(sst3d_yshift)[:, :, 0] / y_dist  # gradient in °C/km

    dSST = np.sqrt((dSST_dx**2) + (dSST_dy**2))
    dSST[dSST > 0.5] = 'nan'

    inshape_dSST = np.ma.array(sst * np.nan)
    inshape_dSST[1:-1, 1:-1] = dSST

    return inshape_dSST


def min2decdeg(lon_or_lat):
    lon_or_lat = np.modf(lon_or_lat)[1] + np.modf(lon_or_lat)[0] / 0.6
    return lon_or_lat


def tsappen(**kwargs):
    '''
    add potential density contours to the current axis
    '''

    ax = kwargs.get('axis', plt.gca())
    pref = kwargs.get('pref', 0)
    levels = kwargs.get('levels', np.arange(20, 31))
    colors = kwargs.get('colors', 'k')

    keys = ['axis', 'pref', 'levels', 'colors']
    for key in keys:
        if key in kwargs:
            kwargs.pop(key)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xax = np.arange(np.min(xlim)-0.01, np.max(xlim)+0.01, 0.01)
    yax = np.arange(np.min(ylim)-0.01, np.max(ylim)+0.01, 0.01)
    sa, t = np.meshgrid(xax, yax)

    # pden = sw.pden(x, y, pref, 0)-1000
    pden = gsw.pot_rho_t_exact(sa, t, pref, 0)-1000
    c = plt.contour(sa, t, pden, levels, colors=colors, **kwargs)
    plt.clabel(c, fmt='%2.1f')


def tsappen_ct(p, **kwargs):
    '''
    add potential density contours to the current axis
    '''

    ax = kwargs.get('axis', plt.gca())
    pref = kwargs.get('pref', 0)
    levels = kwargs.get('levels', np.arange(20, 31))
    colors = kwargs.get('colors', 'k')

    keys = ['axis', 'pref', 'levels', 'colors']
    for key in keys:
        if key in kwargs:
            kwargs.pop(key)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xax = np.arange(np.min(xlim)-0.01, np.max(xlim)+0.01, 0.01)
    yax = np.arange(np.min(ylim)-0.01, np.max(ylim)+0.01, 0.01)
    sa, ct = np.meshgrid(xax, yax)

    # pden = sw.pden(x, y, pref, 0)-1000
    t = gsw.t_from_CT(sa, ct, p)
    pden = gsw.pot_rho_t_exact(sa, t, pref, 0)-1000
    c = plt.contour(sa, ct, pden, levels, colors=colors, **kwargs)
    plt.clabel(c, fmt='%2.1f')

