import numpy as np
import pandas as pd
from numba import jit, prange
import os
import sys

os.chdir("/Users/nokni/work/MHDTurbPy/")

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func

def angle_between_vectors(V, B):
    """
    This function calculates the angle between two vectors.

    Parameters:
    V (ndarray): A 2D numpy array representing the first vector.
    B (ndarray): A 2D numpy array representing the second vector.

    Returns:
    ndarray    : A 1D numpy array representing the angles in degrees between the two input vectors.
    """
    V_norm = np.linalg.norm(V, axis=1, keepdims=True).T[0]
    B_norm = np.linalg.norm(B, axis=1, keepdims=True).T[0]
    dot_product = (V * B).sum(axis=1)
    angle = np.arccos(dot_product / (V_norm * B_norm))/ np.pi * 180
    
    # Restrict angles to [0, 90] 
    angle[angle>90] = 180 - angle[angle>90]
    return angle 


def perp_vector(a, b):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = b / np.linalg.norm(b, axis=1, keepdims=True)
    proj = np.sum((a * b_unit), axis=1, keepdims=True)* b_unit
    perp = a - proj
    return perp

def shifted_df_calcs(B,  lag_coefs, coefs):
    """
    This function calculates the shifted dataframe.

    Parameters:
    B (pandas.DataFrame) : The input dataframe.
    lag_coefs (list)     : A list of integers representing the lags.
    coefs (list)         : A list of coefficients for the calculation.

    Returns:
    ndarray              : A 2D numpy array representing the result of the calculation.
    """
    return pd.DataFrame(np.add.reduce([x*B.shift(y) for x, y in zip(coefs, lag_coefs)]),
                        index=B.index, columns=B.columns).values

def fast_unit_vec(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def local_structure_function(
                             B,
                             V,
                             tau,
                             return_unit_vecs  = False,
                             five_points_sfunc = True
                            ):
    '''
    Parameters:
    B (pandas dataframe)        : The magnetic field data
    V (pandas dataframe)        : The solar wind velocity data
    tau (int)                   : The time lag
    return_unit_vecs (bool)     : Return unit vectors if True, default is False
    five_points_sfunc (bool)    : Use five point structure function if True, default is True

    Returns:
    dB (numpy array)            : The fluctuation of the magnetic field
    VBangle (numpy array)       : The angle between local magnetic field and solar wind velocity
    Phiangle (numpy array)      : The angle between local solar wind velocity perpendicular to local magnetic field and the fluctuation of the magnetic field
    dB_l_perp_hat (numpy array) : Unit vector of the fluctuation of the magnetic field perpendicular to the local magnetic field
    B_l_hat (numpy array)       : Unit vector of the local magnetic field
    B_perp_2_hat (numpy array)  : Unit vector perpendicular to both the fluctuation of the magnetic field and the local magnetic field
    V_l_hat (numpy array)       : Unit vector of the local velocity field
    '''
    # added five_point Structure functions
    if five_points_sfunc:
        # define coefs for loc fields
        coefs_loc     = np.array([1, 4, 6, 4, 1])/16
        lag_coefs_loc = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)

        # define coefs for fluctuations
        coefs_db     = np.array([1, -4, +6, -4, 1])/np.sqrt(35)
        lag_coefs_db = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)
        
        #Compute the fluctuation
        dB           = shifted_df_calcs(B, lag_coefs_db, coefs_db )

        # Estimate local B
        B_l          = shifted_df_calcs(B, lag_coefs_loc, coefs_loc)

        # Estimate local Vsw
        V_l          = shifted_df_calcs(V, lag_coefs_loc, coefs_loc)        
    else:
        #Compute the fluctuation
        dB           = (B.iloc[:-tau].values - B.iloc[tau:].values)

        # Estimate local B
        B_l          = (B.iloc[:-tau].values + B.iloc[tau:].values)/2

        # Estimate local Vsw
        V_l          = (V.iloc[:-tau].values + V.iloc[tau:].values)/2
    
    # Estimate local perpendicular displacement direction
    dB_l_perp        = np.cross(B_l, np.cross(dB, B_l))
 

    #  Estimate the component of the solar wind velocity perpendicular to Blocal
    V_l_perp         = perp_vector(V_l, B_l)

    # Estimate angles
    VBangle          = angle_between_vectors(V_l, B_l)
    Phiangle         = angle_between_vectors(V_l_perp, dB_l_perp )
    
    if return_unit_vecs:

        # Estimate unit vectors
        dB_l_perp_hat = fast_unit_vec(dB_l_perp)
        B_l_hat       = fast_unit_vec(B_l)
        B_perp_2_hat  = np.cross(B_l_hat, dB_l_perp_hat)
        V_l_hat       = None  # V_l / np.linalg.norm(V_l, axis=1, keepdims=True)#.T[0]

    else:
        dB_l_perp_hat = None
        B_l_hat       = None
        B_perp_2_hat  = None
        V_l_hat       = None
    return dB, VBangle, Phiangle, dB_l_perp_hat, B_l_hat, B_perp_2_hat, V_l_hat



@jit( parallel =True,  nopython=True)
def structure_functions_3D(
                           indices, 
                           qorder,
                           mat
                          ):
    """
    Parameters:
    indices (int array)  :  Indices of the data to be processed.
    qorder (int array)   :  Orders of the structure functions to be calculated.
    mat (2D array)       :  Data matrix with 3 columns for 3D field components.

    Returns:
    result (float array) : Structure functions estimated.
    """
    
    # initiate arrays
    result = np.zeros(len(qorder))
    

    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]

    # Estimate sfuncs
    dB = np.sqrt((ar[indices])**2 + 
                 (at[indices])**2 + 
                 (an[indices])**2)

    for i in prange(len(qorder)):   
        result[i] = np.nanmean(np.abs(dB)**qorder[i])
    return list(result)

def estimate_pdfs_3D(
                     indices,
                     mat
                    ):
    """
    A function to estimate probability density functions for 3D field.

    Parameters:
    indices (int array): Indices of the data to be processed.
    mat (2D array): Data matrix with 3 columns for 3D field components.

    Returns:
    result (dict): A dictionary containing the estimated PDFs for each component.
    """
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]

    xPDF_ar, yPDF_ar, _,_ = func.pdf(ar[indices], 45, False, True,scott_rule =False)
    xPDF_at, yPDF_at, _,_ = func.pdf(at[indices], 45, False, True,scott_rule =False)
    xPDF_an, yPDF_an, _,_ = func.pdf(at[indices], 45, False, True,scott_rule =False)

    return {'ar': [xPDF_ar, yPDF_ar], 'at': [xPDF_at, yPDF_at], 'an': [xPDF_an, yPDF_an] }

@jit(nopython=True, parallel=True)
def fast_dot_product(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@jit( nopython=True)
def save_flucs(
               indices,
               mat
              ):
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]
    
    # initiate arrays
    # result = np.zeros(len(qorder))
    
    # Estimate flucs for each component
    d_Br = ar[indices]
    d_Bt = at[indices]
    d_Bn = an[indices]
    return d_Br, d_Bt, d_Bn

def estimate_3D_sfuncs(
                       B,
                       V,
                       dt,
                       Vsw,
                       di, 
                       conditions,
                       qorder,
                       tau_values,
                       estimate_PDFS       = False,
                       return_unit_vecs    = False,
                       five_points_sfuncs  = True,
                       return_coefs        = False,
                       only_general        = False,
                       theta_thresh_gen    = 0,
                       phi_thresh_gen      = 0
                      ):
    """
    Estimate the 3D structure functions for the data given in `B` and `V`

    Parameters
    ----------
    B: array
        magnetic field
    V: array
        velocity field
    dt: float
        time step
    Vsw: float
        solar wind speed
    di: float
        ion inertial length
    conditions: dict
        conditions for each structure function
    qorder: array
        order of the structure function
    tau_values: array
        time lags for the structure function
    estimate_PDFS: bool, optional
        whether to estimate the PDFs for each structure function, default False
    return_unit_vecs: bool, optional
        whether to return the unit vectors, default False
    five_points_sfuncs: bool, optional
        whether to use the 5 point stencil, default True
    return_coefs: bool, optional
        whether to return raw fluctuations or the estimated SF's, default False

    Returns
    -------
    l_di: array
        x values in di
    sf_ell_perp.T: array
        transposed sf_ell_perp
    sf_Ell_perp.T: array
        transposed sf_Ell_perp
    sf_ell_par.T: array
        transposed sf_ell_par
    sf_overall.T: array
        transposed sf_overall
    PDF_dict: dict
        a dictionary with the PDFs for each structure function and overall
    """

    #init conditions
    sf_ell_perp_conds       = conditions['ell_perp']
    sf_Ell_perp_conds       = conditions['Ell_perp']
    sf_ell_par_conds        = conditions['ell_par']
    sf_ell_par_rest_conds   = conditions['ell_par_rest']

    # Initialize arrays
    sf_ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_Ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par              = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par_rest         = np.zeros(( len(tau_values), len(qorder)))
    sf_overall              = np.zeros(( len(tau_values), len(qorder)))
       
    # Initialize dictionaries    
    thetas                  = {}
    phis                    = {}
    
    if return_coefs:
        dBell_perpR = {};   dBEll_perpR = {};  dBell_parR = {}; dBell_par_restR = {}; dB_all_R = {};  
        dBell_perpT = {};   dBEll_perpT = {};  dBell_parT = {}; dBell_par_restT = {}; dB_all_T = {};
        dBell_perpN = {};   dBEll_perpN = {};  dBell_parN = {}; dBell_par_restN = {}; dB_all_N = {};
        

    # Run main loop
    for jj, tau_value in enumerate(tau_values):
        
        # Do the decomposition
        dB, VBangle, Phiangle, dB_l_perp_hat, B_l_hat, B_perp_2_hat,_      = local_structure_function(
                                                                                                       B,
                                                                                                       V,
                                                                                                       int(tau_value), 
                                                                                                       return_unit_vecs,
                                                                                                       five_points_sfunc = five_points_sfuncs
                                                                                                       )

        thetas[str(jj)] = VBangle
        phis[str(jj)]   = Phiangle   
        
        if only_general        == False:

            # for sf_ell_perp

            indices                      = np.where((VBangle>sf_ell_perp_conds['theta']) & (Phiangle>sf_ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs(indices, dB)

                dBell_perpR[str(jj)]     =  d_Br
                dBell_perpT[str(jj)]     =  d_Bt
                dBell_perpN[str(jj)]     =  d_Bn

            else:
                sf_ell_perp[jj, :]       = structure_functions_3D(indices, qorder, dB)

            if estimate_PDFS:
                PDF_ell_perp             = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_perp = None

            # for sf_Ell_perp
            indices                      = np.where((VBangle>sf_Ell_perp_conds['theta']) & (Phiangle<sf_Ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs(indices, dB)

                dBEll_perpR[str(jj)]     =  d_Br
                dBEll_perpT[str(jj)]     =  d_Bt
                dBEll_perpN[str(jj)]     =  d_Bn
            else:
                sf_Ell_perp[jj, :]       = structure_functions_3D(indices, qorder, dB)

            if estimate_PDFS:        
                PDF_Ell_perp             = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_Ell_perp = None

            # for sf_ell_par
            indices                      = np.where((VBangle<sf_ell_par_conds['theta']) & (Phiangle<sf_ell_par_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn = save_flucs(indices, dB)

                dBell_parR[str(jj)]      =  d_Br
                dBell_parT[str(jj)]      =  d_Bt
                dBell_parN[str(jj)]      =  d_Bn
            else:
                sf_ell_par[jj, :]        = structure_functions_3D(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par              = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par = None

            # for sf_ell_par_restricted
            indices                      = np.where((VBangle<sf_ell_par_rest_conds['theta']) & (Phiangle<sf_ell_par_rest_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs(indices, dB)

                dBell_par_restR[str(jj)] =  d_Br
                dBell_par_restT[str(jj)] =  d_Bt
                dBell_par_restN[str(jj)] =  d_Bn
            else:
                sf_ell_par_rest[jj, :]   = structure_functions_3D(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par_rest         = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par_rest = None 
        else:

            # for sf general
            indices                      = np.where((VBangle>theta_thresh_gen) & (Phiangle>phi_thresh_gen))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs(indices, dB)

                dB_all_R[str(jj)]        =  d_Br
                dB_all_T[str(jj)]        =  d_Bt
                dB_all_N[str(jj)]        =  d_Bn
            else:
                sf_overall[jj, :]        = structure_functions_3D(indices, qorder, dB) 

            if estimate_PDFS:
                PDF_all                  = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_all = None    

    # Also estimate x values in di
    l_di    = (tau_values*dt*Vsw)/di
    
    # Return fluctuations
    if  return_coefs:
        if only_general:
            flucts = {
                       'ell_all'          :  {'R': dB_all_R,         'T' : dB_all_T,          'N': dB_all_N    },               
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
                       'di'                :  di,
                       'dt'                :  dt
                     }            
        else:
            flucts = {
                       'ell_perp'          :  {'R': dBell_perpR,      'T' : dBell_perpT,       'N': dBell_perpN },
                       'Ell_perp'          :  {'R': dBEll_perpR,      'T' : dBEll_perpT,       'N': dBEll_perpN },
                       'ell_par'           :  {'R': dBell_parR,       'T' : dBell_parT,        'N': dBell_parN  },
                       'ell_par_rest'      :  {'R': dBell_par_restR,  'T' : dBell_par_restT,   'N': dBell_par_restN  },
                       #'ell_all'          :  {'R': dB_all_R,         'T' : dB_all_T,          'N': dB_all_N    },               
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
                       'di'                :  di,
                       'dt'                :  dt
                     }
    else:
        flucts = None
    
    Sfunctions     = {
                      'ell_perp'     : sf_ell_perp.T,
                      'Ell_perp'     : sf_Ell_perp.T,
                      'ell_par'      : sf_ell_par.T,
                      'ell_par_rest' : sf_ell_par_rest.T,
                      'ell_overall'  : sf_overall.T
    }
    

    try:
        PDFs            =  {
                        'All'          : PDF_all,
                        'ell_par'      : PDF_ell_par,
                        'ell_par_rest' : PDF_ell_par_rest,
                        'ell_perp'     : PDF_Ell_perp,
                        'ell_perp'     : PDF_ell_perp
       }
    except:
        PDFs            = None        
    
    return thetas, phis,  flucts, l_di, Sfunctions, PDFs