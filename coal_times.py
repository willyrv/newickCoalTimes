import re
import numpy as np
import os

def get_branching_times(text_newick):
    """
    Parses the newick tree format and returns the branching times
    starting from the leaves
    """
    branching_times = [0 for i in range(text_newick.count(","))]
    i = 0
    pattern = ":[.0-9]+"
    p = re.compile(pattern)
    current_text = text_newick
    end_pos = current_text.find(",")
    while (end_pos != -1):
        sub_string = current_text[:end_pos]
        partial_branch_length = p.findall(sub_string)
        partial_times = [float(t[1:]) for t in partial_branch_length]
        branching_times[i] = sum(partial_times)
        i +=1
        current_text = current_text[end_pos+1:]
        end_pos = current_text.find(",")
    return(branching_times)

def get_coalescence_times(ms_results):
    """
    Parses the ms results (with the -T option for tree output in newick format)
    and returns a numpy.ndarray with n lines and m columns, where n is the number
    of genes (independent observations of coalescence times). And m is the sample
    size, i.e. column k contains the observed values of T_k
    """
    newick_pattern = "[().:,0-9]+;"
    p = re.compile(newick_pattern)
    branching_times_array = [[0] + get_branching_times(t) for t in p.findall(ms_results)]
    branching_times = np.array(branching_times_array)
    branching_times = np.sort(branching_times)
    coalescence_times = branching_times[:, 1:] - branching_times[:, :-1]
    return(coalescence_times)

def simulate_coalescence_times(ms_cmd):
    """
    For a given ms command, returns a numpy.ndarray with the observations of 
    the coalescence times. For a sample size of n and m independent observations
    (m independent loci) returns a numpy.ndarray with m lines and n columns. Each
    columns k has the observations of T_k.
    """
    ms_results = os.popen(ms_cmd).read()
    return(get_coalescence_times(ms_results))

def compute_empirical_dist(obs, x_vector=False):
    # This method computes the empirical distribution given the
    # observations.
    # The functions are evaluated in the x_vector parameter
    # by default x_vector is computed as a function of the data
    # by default the differences 'dx' are a vector 

    if type(x_vector) == bool:
        actual_x_vector = np.arange(0, max(obs)+0.1, 0.1) 

    elif x_vector[-1]<=max(obs): # extend the vector to cover all the data
        actual_x_vector = list(x_vector)
        actual_x_vector.append(max(obs))
        actual_x_vector = np.array(x_vector)
    else:
        actual_x_vector = np.array(x_vector)
        
    actual_x_vector[0] = 0 # The first element of actual_x_vector should be 0
    
    half_dx = np.true_divide(actual_x_vector[1:]-actual_x_vector[:-1], 2)
    # Computes the cumulative distribution and the distribution
    x_vector_shift = actual_x_vector[:-1] + half_dx
    x_vector_shift = np.array([0] + list(x_vector_shift) + 
                                [actual_x_vector[-1]+half_dx[-1]])
    
    counts = np.histogram(obs, bins = actual_x_vector)[0]
    counts_shift = np.histogram(obs, bins = x_vector_shift)[0]
    
    cdf_x = counts.cumsum()
    cdf_x = np.array([0]+list(cdf_x))
    
    # now we compute the pdf (the derivative of the cdf)
    dy_shift = counts_shift
    dx_shift = x_vector_shift[1:] - x_vector_shift[:-1]
    pdf_obs_x = np.true_divide(dy_shift, dx_shift)

    return (cdf_x, pdf_obs_x)

