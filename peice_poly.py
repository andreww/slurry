#!/usr/bin/env python
# coding=utf8
#
# NOTE: this is extracted from AMW's prem4derg package
#       updates should be copied to that. At some poiny
#       that module will be released via PyPi and this
#       can be declared as a dependancy.
#
"""
Peicewise polynomials like PREM

"""
import numba
import numpy as np

class PeicewisePolynomial(object):
    """
    Peicewise Polynomials a different way
    
    The SciPy PPoly class defines a function from
    polynomials with coefficents c and breakpoints x
    evaluated at a point xp thus:
    
       S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))
    
    This is not helpful for PREM, so we create a new class defining
    the function:
    
       S = sum(c[m, i] * (xp)**(k-m) for m in range(k+1))
      
    Note some important differences between this and PPoly!
    """
    
    def __init__(self, c, x):
        assert len(x.shape)==1, "breakpoints must be 1D"
        self.breakpoints = x 
        if len(c.shape)==1:
            c = np.expand_dims(c, axis=1)
            c = np.append(c,np.zeros_like(c), axis=1)
        assert len(c.shape)==2, "breakpoints must be 2D"
        self.coeffs = c
        
        
    def __call__(self, xp, break_down=False):
        if np.ndim(xp) == 0:
            value = _evaluate_ppoly(xp, self.coeffs, self.breakpoints, break_down)
        else:
            value = _evaluate_ppoly_array(xp, self.coeffs, self.breakpoints, break_down)
        
        return value        
        
        
    def _evaluate_at_point(self, x, break_down=False):
        """
        Evaluate peicewise polynomal at point x
        """
        coef = self._get_coefs(x, break_down)
        value = self._evaluate_polynomial(x, coef)
        return value
        
        
    def _evaluate_polynomial(self, x, coef):
        value = 0
        for i, c in enumerate(coef):
            value = value + c * x**i
        return value
        
        
    def _get_coefs(self, x, break_down=False):
        """
        Return coefs at x
        
        If x falls on a breakpoint, we take the coeffecents from 
        'above' the breakpoint. Unless break_down is True, in which
        case we take the coeffecents from 'below'
        """
        if x == self.breakpoints[-1]:
            # We use the last coefficents for the outside point
            return self.coeffs[-1,:]
        if break_down:
            for i in range(self.breakpoints.size):
                if ((x > self.breakpoints[i]) and 
                              (x <= self.breakpoints[i+1])):
                    return self.coeffs[i,:]
        else:
            for i in range(self.breakpoints.size):
                if ((x >= self.breakpoints[i]) and 
                               (x < self.breakpoints[i+1])):
                    return self.coeffs[i,:]
        
        return None
    
    
    def derivative(self):
    
        deriv_breakpoints = self.breakpoints
        deriv_coeffs = np.zeros((self.coeffs.shape[0],
                                 self.coeffs.shape[1]-1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                if i == 0:
                    continue # Throw away term for x**0
                deriv_coeffs[seg,i-1] = self.coeffs[seg,i]*i
                
        deriv = PeicewisePolynomial(deriv_coeffs, deriv_breakpoints)
        return deriv
    
    def antiderivative(self):
        
        antideriv_breakpoints = self.breakpoints
        antideriv_coeffs = np.zeros((self.coeffs.shape[0],
                                     self.coeffs.shape[1]+1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                antideriv_coeffs[seg,i+1] = self.coeffs[seg,i]/(i+1)
                
        antideriv = PeicewisePolynomial(antideriv_coeffs, 
                                        antideriv_breakpoints)
        return antideriv
    
    
    def integrate(self, a, b):
        
        #antiderivative = self.antiderivative()
        integral = _integrate_ppoly_from_antideriv(a, b, self.coeffs, self.breakpoints)
        return integral
    
    
    def mult(self, other):
        assert self.coeffs.shape[0] == other.coeffs.shape[0], \
                                     'different number of breakpoints'
        mult_breakpoints = self.breakpoints
        mult_coefs = np.zeros((self.coeffs.shape[0],
                               self.coeffs.shape[1]+other.coeffs.shape[1]))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                for j in range(other.coeffs.shape[1]):
                    mult_coefs[seg,i+j] = mult_coefs[seg,i+j] + \
                                 self.coeffs[seg,i] * other.coeffs[seg,j]
                    
        mult_poly = PeicewisePolynomial(mult_coefs, mult_breakpoints)
        return mult_poly

    
@numba.jit(nopython=True)
def _integrate_ppoly_from_antideriv(a, b, coeffs, bps):
    integral = 0
    lower_bound = a
    for bpi, bp in enumerate(bps):
        if bp > lower_bound:
            if bps[bpi] >= b:
                # Just the one segment left - add it and end
                integral = integral + (_evaluate_ppoly(b, coeffs, bps, True) - 
                                       _evaluate_ppoly(lower_bound, coeffs, bps, False))
                break
            else:
                # segment from lower bound to bp
                # add it, increment lower_bound and contiue
                integral = integral + (_evaluate_ppoly(bp, coeffs, bps, True) - 
                                       _evaluate_ppoly(lower_bound, coeffs, bps, False))
                lower_bound = bp

    return integral
    
@numba.jit(nopython=True)
def _evaluate_ppoly_array(xs, coeffs, bps, break_down):   

    values = np.zeros_like(xs)
    for i in range(xs.size):
        values[i] = _evaluate_ppoly(xs[i], coeffs, bps, break_down)
    return values
    
    
@numba.jit(nopython=True)
def _evaluate_ppoly(x, coeffs, bps, break_down): 
     
    # Get the poly coeffs at x
    coef = None
    if x == bps[-1]:
        # We use the last coefficents for the outside point
        coef = coeffs[-1,:]
    elif x == bps[0]:
        coef = coeffs[0,:]
    elif break_down:
        for i in range(bps.size):
            if ((x > bps[i]) and (x <= bps[i+1])):
                coef = coeffs[i,:]
                break
    else:
        for i in range(bps.size):
            if ((x >= bps[i]) and (x < bps[i+1])):
                coef = coeffs[i,:]
                break
        
    if coef is None:
        print("Failed to find coef for ppoly for x, with coeffs, bps, brak_down:")
        print(x)
        print(coeffs)
        print(bps)
        print(break_down)
        assert False, "Failed to find coef"
                         
    # Now evaluate at x
    value = 0
    for i, c in enumerate(coef):
        value = value + c * x**i
        
    return value
