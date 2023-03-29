import numpy as np
import pandas as pd
from numba import jit, prange
import os
import sys

os.chdir("/Users/nokni/work/MHDTurbPy/")

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func
sys.path.insert(1, os.path.join(os.getcwd(), 'functions/modwt/wmtsa'))
import modwt 

def estimate_vec_magnitude_MODWT(xvector):
    """
    Estimate the magnitude of the input vector.

    Parameters:
        xvec (numpy.array): A numpy array representing the input vector.

    Returns:
        numpy.array: A numpy array containing the magnitudes of the input vector.

    """

    return np.linalg.norm(xvector, axis=1,  keepdims=True)

def angle_between_vectors_MODWT(V,
                          B,
                          return_denom  = False,
                          restrict_2_90 = False):
                    
    """
    Calculate the angle between two vectors.

    Args:
        V (np.ndarray)                : A 2D numpy array representing the first vector.
        B (np.ndarray)                : A 2D numpy array representing the second vector.
        return_denom (bool, optional) : Whether to return the denominator components.
        restrict_2_90(bool, optional) : Restrict angles to 0-90
            Defaults to False.

    Returns:
        np.ndarray                    : A 1D numpy array representing the angles in degrees between the two input vectors.
        tuple                         : A tuple containing the angle, dot product, V_norm, and B_norm (if denom is True).
    """
    
    V_norm      = estimate_vec_magnitude_MODWT(V).T[0]
    B_norm      = estimate_vec_magnitude_MODWT(B).T[0]
    
    
    dot_product = (V * B).sum(axis=1)
 
    if restrict_2_90:
        angle       = np.arccos(np.abs(dot_product) / (V_norm * B_norm)) / np.pi * 180
    else:
        angle       = np.arccos(dot_product / (V_norm * B_norm)) / np.pi * 180       
        
    if return_denom:
        return angle, dot_product, V_norm, B_norm
    else:
        return angle


def perp_vector_MODWT(a, b):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = b / estimate_vec_magnitude_MODWT(b)
    proj = (np.sum((a * b_unit), axis=1, keepdims=True))* b_unit
    perp = a - proj
    return perp


# def shifted_df_calcs(B,  lag_coefs, coefs):
#     """
#     This function calculates the shifted dataframe.

#     Parameters:
#     B (pandas.DataFrame) : The input dataframe.
#     lag_coefs (list)     : A list of integers representing the lags.
#     coefs (list)         : A list of coefficients for the calculation.

#     Returns:
#     ndarray              : A 2D numpy array representing the result of the calculation.
#     """
#     return pd.DataFrame(np.add.reduce([x*B.shift(y) for x, y in zip(coefs, lag_coefs)]),
#                         index=B.index, columns=B.columns).values


def mag_of_projection_ells_MODWT(l_vector,
                                 B_l_vector, 
                                 db_perp_vector_MODWT
                                ):
    
    # estimate unit vector in parallel and displacement dir
    B_l_vector     = B_l_vector/estimate_vec_magnitude_MODWT(B_l_vector)
    db_perp_vector_MODWT = db_perp_vector_MODWT/estimate_vec_magnitude_MODWT(db_perp_vector_MODWT)
    
    # estimate unit vector in pependicular by cross product
    b_perp_vector_MODWT  = np.cross(B_l_vector, db_perp_vector_MODWT)
    
    # Calculate dot product in-place
    l_ell     = np.abs(np.nansum(l_vector* B_l_vector, axis=1))
    l_xi      = np.abs(np.nansum(l_vector* db_perp_vector_MODWT, axis=1))
    l_lambda  = np.abs(np.nansum(l_vector* b_perp_vector_MODWT, axis=1))

    return l_ell, l_xi, l_lambda

def PSD_anis_MODWT(coeffs, indices,  iterration, dt):
    coeff = 2**(iterration+1) * dt
    return coeff*(np.nanmean(coeffs.T[0][indices]**2) + np.nanmean(coeffs.T[1][indices]**2) + np.nanmean(coeffs.T[2][indices]**2))


def local_structure_function_MODWT( 
                                  dB,
                                  B_l,
                                  N_l,
                                  V_l,
                                  tau,
                                  dt,
                                  return_unit_vecs         = False,
                                  estimate_alignment_angle = False,
                                  return_mag_align_correl  = False
                            ):
    '''
    Parameters:
    B (pandas dataframe)              : The magnetic field data
    V (pandas dataframe)              : The solar wind velocity data
    tau (int)                         : The time lag
    return_unit_vecs (bool)           : Return unit vectors if True, default is False
    five_points_sfunc (bool)          : Use five point structure function if True, default is True
    estimate_alignment_angle (bool)   : Wether to estimate the alignment angle (Using several different methods)

    Returns:
    dB (numpy array)                  : The fluctuation of the magnetic field
    VBangle (numpy array)             : The angle between local magnetic field and solar wind velocity
    Phiangle (numpy array)            : The angle between local solar wind velocity perpendicular to local magnetic field and the fluctuation of the magnetic field
    dB_perp_hat (numpy array)         : Unit vector of the fluctuation of the magnetic field perpendicular to the local magnetic field
    B_l_hat (numpy array)             : Unit vector of the local magnetic field
    B_perp_2_hat (numpy array)        : Unit vector perpendicular to both the fluctuation of the magnetic field and the local magnetic field
    V_l_hat (numpy array)             : Unit vector of the local
    '''
 

    # Estimate local perpendicular displacement direction
    dB_perp               = perp_vector_MODWT(dB, B_l)

    #Estimate l vector
    l_vec                 = V_l*tau*dt

    # Estrimate l's in three directions
    l_ell, l_xi, l_lambda = mag_of_projection_ells_MODWT(l_vec, B_l, dB_perp)

    #  Estimate the component l perpendicular to Blocal
    l_perp                = perp_vector_MODWT(l_vec, B_l)

    # Estimate angles needed for 3D decomposition
    VBangle               = angle_between_vectors_MODWT(l_vec, B_l, restrict_2_90 = True)
    Phiangle              = angle_between_vectors_MODWT(l_perp, dB_perp,  restrict_2_90 = True)

    # Create empty dictionaries
    unit_vecs             = {}
    align_angles_vb       = {}
    align_angles_zpm      = {}                         
    
    if estimate_alignment_angle:
        
        # Constant to normalize mag field in vel units
        kinet_normal     = 1e-15 / np.sqrt(mu0 * N_l * m_p)

        # Kinetic normalization of magnetic field
        dva_perp         =  dB_perp*kinet_normal

        # We need the perpendicular component of the fluctuations
        du_perp          = perp_vector_MODWT(du, B_l)
        
        # Sign of  background Br
        signB            = - np.sign(B_l.T[0])
        
        # Estimate fluctuations in Elssaser variables
        dzp_perp         = du_perp + (np.array(signB)*dva_perp.T).T
        dzm_perp         = du_perp - (np.array(signB)*dva_perp.T).T
        
        
        #Estimate magnitudes,  angles in three different ways          
        ub_results = est_alignment_angles_MODWT(du_perp, 
                                          dva_perp,
                                          return_mag_align_correl = return_mag_align_correl)
    
        zpm_results = est_alignment_angles_MODWT(dzp_perp, 
                                           dzm_perp,
                                           return_mag_align_correl = return_mag_align_correl)   
                               

                               
        # Assign values for va, v, z+, z-
        sigma_r_mean, sigma_r_median, sins_ub,  v_mag, va_mag, reg_align_angle_sin_ub, polar_int_angle_ub, weighted_sins_ub       = ub_results
        sigma_c_mean, sigma_c_median, sins_zpm, zp_mag, zm_mag, reg_align_angle_sin_zpm, polar_int_angle_zpm, weighted_sins_zpm   = zpm_results   
                               

        align_angles_vb      = {     
                                     'sig_r_mean'        : sigma_r_mean,
                                     'sig_r_median'      : sigma_r_median,
                                     'reg_angle'         : reg_align_angle_sin_ub,
                                     'polar_inter_angle' : polar_int_angle_ub,            
                                     'weighted_angle'    : weighted_sins_ub,
                                     'v_mag'             : v_mag,
                                     'va_mag'            : va_mag,
                                     'sins_uva'          : sins_ub,
        }
                               
        align_angles_zpm     = {     
                                     'sig_c_mean'        : sigma_c_mean,
                                     'sig_c_median'      : sigma_c_median,
                                     'reg_angle'         : reg_align_angle_sin_zpm,
                                     'polar_inter_angle' : polar_int_angle_zpm,            
                                     'weighted_angle'    : weighted_sins_zpm,
                                     'zp_mag'            : zp_mag,
                                     'zm_mag'            : zm_mag,
                                     'sins_zpm'          : sins_zpm,
        }

    
    if return_unit_vecs:
        # Estimate unit vectors
        dB_perp_hat = np.array(dB_perp) / np.array(np.linalg.norm(dB_perp, axis=1, keepdims=True))
        B_l_hat       = B_l / np.linalg.norm(B_l, axis=1, keepdims=True)
        B_perp_2_hat  = np.cross(B_l_hat, dB_perp_hat)
        V_l_hat       = V_l / np.linalg.norm(V_l, axis=1, keepdims=True)#.T[0]

    else:
        dB_perp_hat   = None
        B_l_hat       = None
        B_perp_2_hat  = None
        V_l_hat       = None

    return dB, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm
           



@jit( parallel =True,  nopython=True)
def structure_functions_3D_MODWT(indices, qorder,mat):
    """
    Parameters:
    indices (int array)  :  Indices of the data to be processed.
    qorder (int array)   :  Orders of the structure functions to be calculated.
    mat (2D array)       :  Data matrix with 3 columns for 3D field components.

    Returns:
    result (float array) : Structure functions estimated.
    """
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]
    
    # initiate arrays
    result = np.zeros(len(qorder))
    
    # Estimate sfuncs
    dB = np.sqrt((ar[indices])**2 + 
                 (at[indices])**2 + 
                 (an[indices])**2)

    for i in prange(len(qorder)):   
        result[i] = np.nanmean(np.abs(dB**qorder[i]))
    return list(result)

def estimate_pdfs_3D_MODWT(indices, mat):
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



@jit(nopython=True)
def save_flucs_MODWT(
               indices,
               mat, 
               ells
              ):
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]
    
    # initiate arrays
    # result = np.zeros(len(qorder))
    
    # Estimate flucs for each component
    d_Br    = ar[indices]
    d_Bt    = at[indices]
    d_Bn    = an[indices]
    ell_fin = ells[indices]
    return d_Br, d_Bt, d_Bn, ell_fin

def est_alignment_angles_MODWT(
                         xvec,
                         yvec,
                         return_mag_align_correl = False):
    """
    Calculate the sine of the angle between two vectors.

    Parameters:
        xvec (numpy.array): A numpy array representing the first input vector.
        yvec (numpy.array): A numpy array representing the second input vector.

    Returns:
        numpy.array: A numpy array containing the sine values of the angles between the input vectors.
    """
    
    # Estimate cross product of the two vectors
    
    numer          = np.cross(xvec, yvec)

    # Estimate magnitudes of the two vectors:
    xvec_mag       = estimate_vec_magnitude_MODWT(xvec)
    yvec_mag       = estimate_vec_magnitude_MODWT(yvec)

    # Estimate sigma (sigma_r for (δv, δb), sigma_c for (δzp, δz-) )
    sigma_mean           = np.nanmean((xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 ))
    sigma_median         = np.nanmedian((xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 ))   

    # Estimate denominator
    denom          = (xvec_mag*yvec_mag)
    
    # Make sure we dont have inf vals
    numer[np.isinf(numer)] = np.nan
    denom[np.isinf(denom)] = np.nan

    numer          = estimate_vec_magnitude_MODWT(numer)
    denom          = np.abs(denom)


    # Estimate sine of the  two vectors
    sins              = (numer/denom)
    thetas            = np.arcsin(sins)*180/np.pi
    thetas[thetas>90] = 180 -thetas[thetas>90]
    
    # Regular alignment angle
    reg_align_angle_sin = np.nanmean(sins)
    
    # polarization intermittency angle (Beresnyak & Lazarian 2006):
    polar_int_angle = (np.nanmean(numer)/ np.nanmean(denom))   

    # Weighted angles
    weighted_sins  = np.sin(np.nansum(thetas*(denom / np.nansum(denom)))*np.pi/180)
    #weighted_sins  = np.nansum(((sins)*(denom / np.nansum(denom))))
                               
    if return_mag_align_correl== False:
        sins, xvec_mag, yvec_mag = None, None, None
                               
    return sigma_mean, sigma_median, sins, xvec_mag, yvec_mag, reg_align_angle_sin, polar_int_angle, weighted_sins




def estimate_coeffs_background_flucs_MODWT(x, wname):
    

    # Estimate length of timeseries
    sample_length = len(x)
    
    # Estimate MODWT coefficients and weights
    Wj, Vj        = modwt.modwt(x, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    
    # Perform forwards multiresolution analysis obtain 
    # fluctuations (details) and background (approximations) at each level
    Det, Appr     = modwt.imodwt_mra(Wj, Vj)
    
    # It returns a timeseries with length 2x sample_length
    Det, Appr     = Det[:, 0: sample_length],  Appr[ 0: sample_length]
    
    # Reconstruct the approximations at each level using the details
    Approx  = []
    for i in range(len(Det)):
        if i==0:
            Approx.append(Appr)
        else:
            Approx.append(Approx[i-1] + Det[i-1])
    
    # Remove the phase shift in the detail coefficients at each levels 
    Swd, Vjd       = modwt.cir_shift(Wj, Vj, subtract_mean_VJ0t=True)
 
    return Approx[::-1], Det[::-1], Swd

def estimate_V_background_MODWT(x, wname):
    

    # Estimate length of timeseries
    sample_length = len(x)
    
    # Estimate MODWT coefficients and weights
    Wj, Vj   = modwt.modwt(x, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    
    # Perform forwards multiresolution analysis obtain 
    # fluctuations (details) and background (approximations) at each level
    Det, Appr  = modwt.imodwt_mra(Wj, Vj)
    
    # It returns a timeseries with length 2x sample_length
    Det, Appr  = Det[:, 0: sample_length],  Appr[ 0: sample_length]
    
    # Reconstruct the approximations at each level using the details
    Approx  = []
    for i in range(len(Det)):
        if i==0:
            Approx.append(Appr)
        else:
            Approx.append(Approx[i-1] + Det[i-1])

 
    return Approx[::-1]

def estimate_3D_sfuncs_MODWT(
                       J,
                       flucs,
                       freqs,
                       db,
                       Bl,
                       N_l, 
                       Vl,
                       dt,
                       Vsw,
                       di,
                       qorder,
                       conditions,
                       theta_thresh_gen         = 0,
                       phi_thresh_gen           = 0,
                       estimate_PDFS            = False,
                       return_unit_vecs         = False,
                       return_coefs             = False,
                       estimate_alignment_angle = False,
                       return_mag_align_correl  = False,
                       only_general             = False        
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

    # Define lags
    tau_values      = 1/freqs;
    wave_scales     = 2**np.arange(1,J+1);

    # Initialize arrays
    sf_ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_Ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par              = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par_rest         = np.zeros(( len(tau_values), len(qorder)))
    sf_overall              = np.zeros(( len(tau_values), len(qorder)))
    


    # Initialize dictionaries    
    thetas                  = {}
    phis                    = {}
    ub_polar                = []
    ub_reg                  = []
    ub_weighted             = []
    zpm_polar               = []
    zpm_reg                 = []
    zpm_weighted            = []
    u_norms                 = {}
    b_norms                 = {}
    sig_c_mean              = []
    sig_r_mean              = []
    sig_c_median            = []
    sig_r_median            = []

    if return_coefs:
        dBell_perpR = {};   dBEll_perpR = {};  dBell_parR = {}; dBell_par_restR = {}; dB_all_R = {};  
        dBell_perpT = {};   dBEll_perpT = {};  dBell_parT = {}; dBell_par_restT = {}; dB_all_T = {};
        dBell_perpN = {};   dBEll_perpN = {};  dBell_parN = {}; dBell_par_restN = {}; dB_all_N = {};
        lambdas     = {};   xis         = {};  ells       = {}; ells_rest       = {};
        


    # Run main loop
    for jj, tau_value in enumerate(tau_values):
        
        Bls    = np.array([Bl['R'][jj],    Bl['T'][jj],    Bl['N'][jj]]).T
        Vls    = np.array([Vl['R'][jj],    Vl['T'][jj],    Vl['N'][jj]]).T
        dBs    = np.array([db['R'][jj],    db['T'][jj],    db['N'][jj]]).T
        flucts = np.array([flucs['R'][jj], flucs['T'][jj], flucs['N'][jj]]).T  
                                                                                                        
        # Do the decomposition
        dB, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm =  local_structure_function_MODWT(
                                                                                                                                      dBs,
                                                                                                                                      Bls,
                                                                                                                                      N_l,
                                                                                                                                      Vls,
                                                                                                                                      tau_value,
                                                                                                                                      dt,
                                                                                                                                      return_unit_vecs         = return_unit_vecs,
                                                                                                                                      estimate_alignment_angle = estimate_alignment_angle,
                                                                                                                                      return_mag_align_correl  = return_mag_align_correl
                                                                                                                              )


        
        thetas[str(jj)] = VBangle
        phis[str(jj)]   = Phiangle   

        if estimate_alignment_angle:
            # Va, v average angles
            ub_polar.append(align_angles_vb['polar_inter_angle'])
            ub_reg.append(align_angles_vb['reg_angle'])
            ub_weighted.append(align_angles_vb['weighted_angle'])
            sig_r_mean.append(align_angles_vb['sig_r_mean'])
            sig_r_median.append(align_angles_vb['sig_r_median'])            

            # Zp, Zm average angles
            zpm_polar.append(align_angles_zpm['polar_inter_angle'])
            zpm_reg.append(align_angles_zpm['reg_angle'])
            zpm_weighted.append(align_angles_zpm['weighted_angle'])
            sig_c_mean.append(align_angles_zpm['sig_c_mean'])
            sig_c_median.append(align_angles_zpm['sig_c_median'])    
        
        if only_general        == False:

            # for sf_ell_perp
            indices                      = np.where((VBangle>sf_ell_perp_conds['theta']) & (Phiangle>sf_ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_lambda_fin = save_flucs_MODWT(indices, dB, l_lambda)

                dBell_perpR[str(jj)]     =  d_Br
                dBell_perpT[str(jj)]     =  d_Bt
                dBell_perpN[str(jj)]     =  d_Bn
                lambdas[str(jj)]         =  l_lambda_fin

            else:
                sf_ell_perp[jj, :]       = structure_functions_3D_MODWT(indices, qorder, dB)

            if estimate_PDFS:
                PDF_ell_perp             = estimate_pdfs_3D_MODWT(indices,  dB)
            else:
                PDF_ell_perp = None

            # for sf_Ell_perp
            indices                      = np.where((VBangle>sf_Ell_perp_conds['theta']) & (Phiangle<sf_Ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_xi_fin         = save_flucs_MODWT(indices, dB, l_xi)

                dBEll_perpR[str(jj)]     =  d_Br
                dBEll_perpT[str(jj)]     =  d_Bt
                dBEll_perpN[str(jj)]     =  d_Bn
                xis[str(jj)]             =  l_xi_fin
            else:
                sf_Ell_perp[jj, :]       = structure_functions_3D_MODWT(indices, qorder, dB)

            if estimate_PDFS:        
                PDF_Ell_perp             = estimate_pdfs_3D_MODWT(indices,  dB)
            else:
                PDF_Ell_perp = None

            # for sf_ell_par
            indices                      = np.where((VBangle<sf_ell_par_conds['theta']) & (Phiangle<sf_ell_par_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_ell_fin = save_flucs_MODWT(indices, dB, l_ell)

                dBell_parR[str(jj)]      =  d_Br
                dBell_parT[str(jj)]      =  d_Bt
                dBell_parN[str(jj)]      =  d_Bn
                ells[str(jj)]            =  l_ell_fin
            else:
                sf_ell_par[jj, :]        = structure_functions_3D_MODWT(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par              = estimate_pdfs_3D_MODWT(indices,  dB)
            else:
                PDF_ell_par = None

            # for sf_ell_par_restricted
            indices                      = np.where((VBangle<sf_ell_par_rest_conds['theta']) & (Phiangle<sf_ell_par_rest_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_ell_fin_rest    = save_flucs_MODWT(indices, dB, l_ell)

                dBell_par_restR[str(jj)] =  d_Br
                dBell_par_restT[str(jj)] =  d_Bt
                dBell_par_restN[str(jj)] =  d_Bn
                ells_rest[str(jj)]       =  l_ell_fin_rest
            else:
                sf_ell_par_rest[jj, :]   = structure_functions_3D_MODWT(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par_rest         = estimate_pdfs_3D_MODWT(indices,  dB)
            else:
                PDF_ell_par_rest = None 
        else:

            # for sf general
            indices                      = np.where((VBangle>theta_thresh_gen) & (Phiangle>phi_thresh_gen))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs_MODWT(indices, dB)

                dB_all_R[str(jj)]        =  d_Br
                dB_all_T[str(jj)]        =  d_Bt
                dB_all_N[str(jj)]        =  d_Bn
            else:
                sf_overall[jj, :]        = structure_functions_3D_MODWT(indices, qorder, dB) 

            if estimate_PDFS:
                PDF_all                  = estimate_pdfs_3D_MODWT(indices,  dB)
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
                       'dt'                :  dt, 'wave_scales': wave_scales
                     }            
        else:
            flucts = {
                       'ell_perp'          :  {'R': dBell_perpR,      'T' : dBell_perpT,       'N': dBell_perpN,     'lambdas'  : lambdas  },
                       'Ell_perp'          :  {'R': dBEll_perpR,      'T' : dBEll_perpT,       'N': dBEll_perpN,     'xis'      : xis      },
                       'ell_par'           :  {'R': dBell_parR,       'T' : dBell_parT,        'N': dBell_parN ,     'ells'     : ells     },
                       'ell_par_rest'      :  {'R': dBell_par_restR,  'T' : dBell_par_restT,   'N': dBell_par_restN, 'ells_rest': ells_rest},          
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
                       'di'                :  di,
                       'dt'                :  dt , 'wave_scales': wave_scales
                     }
    else:
        flucts = None
    
    Sfunctions     = {
                      'ell_perp'     : sf_ell_perp.T,
                      'Ell_perp'     : sf_Ell_perp.T,
                      'ell_par'      : sf_ell_par.T,
                      'ell_par_rest' : sf_ell_par_rest.T,
                      'ell_overall'  : sf_overall.T, 'wave_scales': wave_scales
    }
    

    try:
        PDFs            =  {
                        'All'          : PDF_all,
                        'ell_par'      : PDF_ell_par,
                        'ell_par_rest' : PDF_ell_par_rest,
                        'ell_perp'     : PDF_Ell_perp,
                        'ell_perp'     : PDF_ell_perp, 'wave_scales': wave_scales
       }
    except:
        PDFs            = None   
    if estimate_alignment_angle:
        overall_align_angles = {
                                'VB' :  {'reg': ub_reg,  'polar':  ub_polar, 'weighted': ub_weighted, 'sig_r_mean': sig_r_mean, 'sig_r_median': sig_r_median},
                                'Zpm':  {'reg': zpm_reg, 'polar': zpm_polar, 'weighted': zpm_weighted,'sig_c_mean': sig_c_mean, 'sig_c_median': sig_c_median}            
                               }
    else:
        overall_align_angles = None           
    
    return thetas, phis,  flucts, l_di, Sfunctions, PDFs, overall_align_angles
