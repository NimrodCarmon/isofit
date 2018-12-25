#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import scipy as s

from common import recursive_replace, eps
from copy import deepcopy
from scipy.linalg import det, norm, pinv, sqrtm, inv, block_diag
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar as min1d
from rt_modtran import ModtranRT
from rt_libradtran import LibRadTranRT
from rt_planetary import PlanetaryRT
from surf import Surface
from surf_multicomp import MultiComponentSurface
from surf_cat import CATSurface
from surf_glint import GlintSurface
from surf_emissive import MixBBSurface
from surf_iop import IOPSurface
from instrument import Instrument


class ForwardModel:

    def __init__(self, config):
        '''ForwardModel contains all the information about how to calculate
         radiance measurements at a specific spectral calibration, given a 
         state vector.  It also manages the distributions of unretrieved, 
         unknown parameters of the state vector (i.e. the S_b and K_b 
         matrices of Rodgers et al.'''

        self.instrument = Instrument(config['instrument'])
        self.n_meas = self.instrument.n_chan

        # Build the radiative transfer model
        if 'modtran_radiative_transfer' in config:
            self.RT = ModtranRT(config['modtran_radiative_transfer'])
        elif 'libradtran_radiative_transfer' in config:
            self.RT = LibRadTranRT(config['libradtran_radiative_transfer'])        
        elif 'planetary_radiative_transfer' in config:
            self.RT = PlanetaryRT(config['planetary_radiative_transfer'])
        else:
            raise ValueError('Must specify a valid radiative transfer model')

        # Build the surface model
        if 'surface' in config:
            self.surface = Surface(config['surface'])
        elif 'multicomponent_surface' in config:
            self.surface = MultiComponentSurface(config['multicomponent_surface'])
        elif 'emissive_surface' in config:
            self.surface = MixBBSurface(config['emissive_surface'])
        elif 'cat_surface' in config:
            self.surface = CATSurface(config['cat_surface'])
        elif 'glint_surface' in config:
            self.surface = GlintSurface(config['glint_surface'])
        elif 'iop_surface' in config:
            self.surface = IOPSurface(config['iop_surface'])
        else:
            raise ValueError('Must specify a valid surface model')

        # Set up passthrough option
        bounds, scale, init_val, statevec = [], [], [], []

        # Build state vector for each part of our forward model
        for name in ['surface', 'RT', 'instrument']:
            obj = getattr(self, name)
            inds = len(statevec) + s.arange(len(obj.statevec), dtype=int)
            setattr(self, 'idx_%s' % name, inds)
            for b in obj.bounds:
                bounds.append(deepcopy(b))
            for c in obj.scale:
                scale.append(deepcopy(c))
            for v in obj.init_val:
                init_val.append(deepcopy(v))
            for v in obj.statevec:
                statevec.append(deepcopy(v))
        self.bounds = tuple(s.array(bounds).T)
        self.scale = s.array(scale)
        self.init_val = s.array(init_val)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        # Capture unmodeled variables
        bvec, bval = [], []
        for name in ['RT', 'instrument', 'surface']:
            obj = getattr(self, name)
            inds = len(bvec) + s.arange(len(obj.bvec), dtype=int)
            setattr(self, '%s_b_inds' % name, inds)
            for b in obj.bval:
                bval.append(deepcopy(b))
            for v in obj.bvec:
                bvec.append(deepcopy(v))
        self.bvec = s.array(bvec)
        self.nbvec = len(self.bvec)
        self.bval = s.array(bval)
        self.Sb = s.diagflat(pow(self.bval, 2))
        return

    def out_of_bounds(self, x):
        """Is state vector inside the bounds?"""
        x_RT = x[self.idx_RT]
        x_surface = x[self.idx_surface]
        bound_lwr = self.bounds[0]
        bound_upr = self.bounds[1]
        return any(x_RT >= (bound_upr[self.idx_RT] - eps*2.0)) or \
            any(x_RT <= (bound_lwr[self.idx_RT] + eps*2.0))

    def xa(self, x, geom):
        """Calculate the prior mean of the state vector (the concatenation
        of state vectors for the surface, Radiative Transfer model, and 
        instrument). Note that the surface prior mean depends on the 
        current state - this is so we can calculate the local prior."""

        x_surface = x[self.idx_surface]
        xa_surface = self.surface.xa(x_surface, geom)
        xa_RT = self.RT.xa()
        xa_instrument = self.instrument.xa()
        return s.concatenate((xa_surface, xa_RT, xa_instrument), axis=0)

    def Sa(self, x, geom):
        """Calculate the prior covariance of the state vector (the concatenation
        of state vectors for the surface and Radiative Transfer model).
        Note that the surface prior depends on the current state - this
        is so we can calculate the local linearized answer."""

        x_surface = x[self.idx_surface]
        Sa_surface = self.surface.Sa(x_surface, geom)[:, :]
        Sa_RT = self.RT.Sa()[:, :]
        Sa_instrument = self.instrument.Sa()[:,:]
        return block_diag(Sa_surface, Sa_RT, Sa_instrument)

    def init(self, meas, geom):
        """Find an initial guess at the state vector.  This currently uses
        traditional (non-iterative, heuristic) atmospheric correction."""

        x = self.init_val.copy()
        x[self.idx_RT] = self.heuristic_atmosphere(x, meas, geom)
        rfl_est, Ls_est, coeffs = self.invert_algebraic(x, meas, geom)

        if not (isinstance(self.surface, MixBBSurface) or 
                isinstance(self.surface, CATSurface)):
            Ls_est = None
        else:
            # modify reflectance and estimate surface emission
            rfl_est = self.surface.conditional_solrfl(rfl_est, geom)
            Ls_est  = self.estimate_Ls(x_RT, rfl_est, meas, geom)
        x[self.idx_surface] = self.surface.fit_params(rfl_est, Ls_est, geom)
        return x 

    def calc_rdn(self, x, geom, rfl=None, Ls=None):
        """Calculate the high-resolution radiance, permitting overrides
        Project to top of atmosphere and translate to radiance"""

        x_surface, x_RT = x[self.idx_surface],  x[self.idx_RT]
        if rfl is None:
            rfl = self.surface.calc_rfl(x_surface, geom)
        if Ls is None:
            Ls = self.surface.calc_Ls(x_surface, geom)
        rfl_hi = self.upsample_surface(rfl)
        Ls_hi  = self.upsample_surface(Ls)
        return self.RT.calc_rdn(x_RT, rfl_hi, Ls_hi, geom)

    def calc_meas(self, x, geom, rfl=None, Ls=None):
        """Calculate the model observation at insttrument wavelengths"""

        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        x_instrument = x[self.idx_instrument]
        rdn_hi = self.calc_rdn(x, geom, rfl, Ls)
        return self.instrument.sample(x_instrument, self.RT.wl, rdn_hi)

    def calc_Ls(self, x, geom):
        """calculate the surface emission."""

        return self.surface.calc_Ls(x[self.idx_surface], geom)

    def calc_rfl(self, x, geom):
        """calculate the surface reflectance"""

        return self.surface.calc_rfl(x[self.idx_surface], geom)

    def calc_lamb(self, x, geom):
        """calculate the Lambertian surface reflectance"""

        return self.surface.calc_lamb(x[self.idx_surface], geom)

    def Seps(self, x, meas, geom):
        """Calculate the total uncertainty of the observation, including
        both the instrument noise and the uncertainty due to unmodeled
        variables. This is the S_epsilon matrix of Rodgers et al."""

        Kb = self.Kb(x, geom)
        Sy = self.instrument.Sy(meas, geom)
        return Sy + Kb.dot(self.Sb).dot(Kb.T)

    def K(self, x, geom):
        """Derivative of observation with respect to state vector. This is 
        the concatenation of jacobians with respect to parameters of the 
        surface and radiative transfer model."""

        # Unpack state vector
        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        x_instrument = x[self.idx_instrument]

        # Get partials of reflectance WRT state, and upsample
        rfl = self.surface.calc_rfl(x_surface, geom)
        drfl_dsurface = self.surface.drfl_dsurface(x_surface, geom)
        rfl_hi = self.upsample_surface(rfl)
        drfl_dsurface_hi = self.upsample_surface(drfl_dsurface.T).T

        # Get partials of emission WRT state, and upsample
        Ls = self.surface.calc_Ls(x_surface, geom)
        dLs_dsurface = self.surface.dLs_dsurface(x_surface, geom)
        Ls_hi = self.upsample_surface(Ls)
        dLs_dsurface_hi = self.upsample_surface(dLs_dsurface.T).T

        # Derivatives of RTM radiance
        drdn_dRT, drdn_dsurface = self.RT.drdn_dRT(x_RT, x_surface, rfl_hi, 
            drfl_dsurface_hi, Ls_hi, dLs_dsurface_hi, geom)
        
        # Derivatives of measurement, avoiding recalculation of rfl, Ls
        dmeas_dsurface = self.instrument.sample(x_instrument, self.RT.wl,
            drdn_dsurface.T).T
        dmeas_dRT = self.instrument.sample(x_instrument, self.RT.wl, drdn_dRT.T).T
        rdn_hi = self.calc_rdn(x, geom, rfl=rfl, Ls=Ls)
        dmeas_dinstrument = self.instrument.dmeas_dinstrument(x_instrument,
            self.RT.wl, rdn_hi)

        # Put it all together
        K = s.zeros((self.n_meas, self.nstate), dtype=float)
        K[:, self.idx_surface] = dmeas_dsurface
        K[:, self.idx_RT] = dmeas_dRT
        K[:, self.idx_instrument] = dmeas_dinstrument
        return K

    def Kb(self, x, geom):
        """Derivative of measurement with respect to unmodeled & unretrieved
        unknown variables, e.g. S_b. This is  the concatenation of jacobians 
        with respect to parameters of the surface, radiative transfer model, 
        and instrument. """

        # Unpack state vector
        # TODO: Add reflectance uncertainty someday
        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        x_instrument = x[self.idx_instrument]

        # Get partials of reflectance and upsample
        rfl = self.surface.calc_rfl(x_surface, geom)
        rfl_hi = self.upsample_surface(rfl)
        Ls = self.surface.calc_Ls(x_surface, geom)
        Ls_hi = self.upsample_surface(Ls)
        rdn_hi = self.calc_rdn(x, geom, rfl=rfl, Ls = Ls)

        drdn_dRTb = self.RT.drdn_dRTb(x_RT, rfl_hi, Ls_hi, geom)
        dmeas_dRTb = self.instrument.sample(x_instrument,self.RT.wl,
                drdn_dRTb.T).T
        dmeas_dinstrumentb = self.instrument.dmeas_dinstrumentb(\
                x_instrument, self.RT.wl, rdn_hi)

        Kb = s.zeros((self.n_meas, self.nbvec), dtype=float)
        #Kb[:, self.surface_b_inds] = dmeas_dsurfaceb
        Kb[:, self.RT_b_inds] = dmeas_dRTb
        Kb[:, self.instrument_b_inds] = dmeas_dinstrumentb
        return Kb

    def summarize(self, x, geom):
        """State vector summary"""
        x_RT = x[self.idx_RT]
        x_surface = x[self.idx_surface]
        return self.surface.summarize(x_surface, geom) + \
            ' ' + self.RT.summarize(x_RT, geom)

    def calibration(self, x):
        """Calculate measured wavelengths and fwhm"""
        x_inst = x[self.idx_instrument]
        return self.instrument.calibration(x_inst)

    def upsample_surface(self, q):
        """Linear interpolation of surface to RT wavelengths"""
        if q.ndim > 1:
            q2 = []
            for qi in q: 
                p = interp1d(self.surface.wl, qi, bounds_error=False,
                    fill_value='extrapolate')
                q2.append(p(self.RT.wl))
            return s.array(q2)
        else:
            p = interp1d(self.surface.wl, q, bounds_error=False, 
                fill_value='extrapolate')
            return p(self.RT.wl)

    def heuristic_atmosphere(self, x, meas, geom):
        '''From a given radiance, estimate atmospheric state with band ratios.
        Used to initialize gradient descent inversions.'''

        # unpack the state vector and determine the calibration
        x_RT = x[self.idx_RT].copy()
        x_instrument = x[self.idx_instrument].copy()
        wl, fwhm = self.instrument.calibration(x_instrument)
        b865 = s.argmin(abs(wl-865))
        b945 = s.argmin(abs(wl-945))
        b1040 = s.argmin(abs(wl-1040))
    
        # Band ratio retrieval of H2O
        for h2oname in ['H2OSTR', 'h2o']:
            if (h2oname in self.RT.lut_names) and \
                    any(self.RT.wl > 850) and any(self.RT.wl < 1050):

                ind_lut = self.RT.lut_names.index(h2oname)
                ind_sv = self.RT.statevec.index(h2oname)
                h2os, ratios = [], []  

                for h2o in self.RT.lut_grids[ind_lut]:

                    # Get Atmospheric terms at high spectral resolution
                    x_RT_new = x_RT.copy()
                    x_RT_new[ind_sv] = h2o
                    rhoatm_hi, sphalb_hi, transm_hi, transup_hi = \
                        self.RT.get(x_RT_new, geom)
                    rhoatm = self.instrument.sample(x_instrument, 
                            self.RT.wl, rhoatm_hi)
                    transm = self.instrument.sample(x_instrument, 
                            self.RT.wl, transm_hi)
                    solar_irr = self.instrument.sample(x_instrument, 
                            self.RT.wl, self.RT.solar_irr)

                    # Assume no surface emission here
                    r = (meas*s.pi/(solar_irr*self.RT.coszen) -
                         rhoatm) / (transm+1e-8)
                    ratios.append((r[b945]*2.0)/(r[b1040]+r[b865]))
                    h2os.append(h2o)

                p = interp1d(h2os, ratios)
                bounds = (h2os[0]+0.001, h2os[-1]-0.001)
                best = min1d(lambda h: abs(1-p(h)),
                             bounds=bounds, method='bounded')
                x_RT[ind_sv] = best.x
        return x_RT

    def invert_algebraic(self, x, meas, geom):
        '''Inverts radiance algebraically to get a reflectance.'''

        # Unpack the state vector
        x_RT = x[self.idx_RT].copy()
        x_surface = x[self.idx_surface].copy()
        x_instrument = x[self.idx_instrument].copy()
        wl, fwhm = self.instrument.calibration(x_instrument)

        # Get atmospheric optical parameters (possibly at high 
        # spectral resolution) and resample them if needed.
        rhoatm_hi, sphalb_hi, transm_hi, transup_hi = \
            self.RT.get(x_RT, geom)
        rhoatm = self.instrument.sample(x_instrument, 
                self.RT.wl, rhoatm_hi)
        transm = self.instrument.sample(x_instrument, 
                self.RT.wl, transm_hi)
        solar_irr = self.instrument.sample(x_instrument, 
                self.RT.wl, self.RT.solar_irr)
        sphalb = self.instrument.sample(x_instrument, 
                self.RT.wl, sphalb_hi)
        transup = self.instrument.sample(x_instrument, 
                self.RT.wl, transup_hi)

        # surface and measured wavelengths may differ.  Calculate
        # the initial emission and subtract from the measurement
        Ls = self.surface.calc_Ls(x_surface, geom)
        Ls_meas = interp1d(self.surface.wl, Ls)(wl)
        rdn_solrfl = meas - (transup * Ls_meas)

        # Now solve for the reflectance at measured wavelengths,
        # and back-translate to surface wavelengths
        rho = rdn_solrfl * s.pi / (self.RT.solar_irr * self.RT.coszen)
        rfl = 1.0 / (transm / (rho - rhoatm) + sphalb)
        rfl_est = interp1d(wl, rfl)(self.surface.wl)
        coeffs = rhoatm, sphalb, transm, self.RT.solar_irr, self.RT.coszen
        return rfl_est, Ls, coeffs

    def estimate_Ls(self, x_RT, rfl, rdn, geom):
        """Estimate the surface emission for a given state vector and 
           reflectance/radiance pair"""

        rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
        rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
        Ls = (rdn - rho/s.pi*(self.solar_irr*self.coszen)) / transup
        return Ls

