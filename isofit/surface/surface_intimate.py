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

import numpy as np
import scipy.linalg

from ..core.common import emissive_radiance, eps
from .surface_multicomp import MultiComponentSurface
from isofit.configs import Config

import pdb
class IntimateSurface(MultiComponentSurface):
    """A model of surface reflectance accounting for single and multiple scattering in 
    the reflectance using a sum of a power series formulation. Creates a nonlinear 
    response in the reflectance"""
        
    def __init__(self, full_config: Config):
        """."""

        config = full_config.forward_model.surface

        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        # Handle additional state vector elements
        ''' the a parameter for intimate mixtures '''
        self.statevec_names.extend(['Areal_Mix_Ratio'])
        self.init.extend([0.99])  # This is overwritten below
        self.scale.extend([0.05])
        self.bounds.extend([[0.6, 1]])
        self.areal_mix_ind = len(self.statevec_names) - 1
        self.n_state = len(self.init)
        self.areal_mix_prior_mu = 0.8
        self.areal_mix_prior_sigma = 0.2
        
        ''' the b parameter for shadow fraction '''
        self.statevec_names.extend(['Shadow_fraction'])
        self.init.extend([0.99])  # This is overwritten below
        self.scale.extend([0.05])
        self.bounds.extend([[0.6, 1]])
        self.shadow_fraction_ind = len(self.statevec_names) - 1
        self.n_state = len(self.init)
        self.shadow_fraction_prior_mu = 0.8
        self.shadow_fraction_prior_sigma = 0.2

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.areal_mix_ind] = self.areal_mix_prior_mu
        mu[self.shadow_fraction_ind] = self.shadow_fraction_prior_mu
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        Cov[self.areal_mix_ind, self.areal_mix_ind] = \
            self.areal_mix_prior_sigma**2
        Cov[self.shadow_fraction_ind, self.shadow_fraction_ind] = \
            self.shadow_fraction_prior_sigma**2
            

        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, find the surface reflectance"""

        x_surface = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        x_surface[self.areal_mix_ind] = self.init[self.areal_mix_ind]
        x_surface[self.shadow_fraction_ind] = self.init[self.shadow_fraction_ind]
        

        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Reflectance. This could be overriden to add (for example)
            specular components"""
        

        return self.calc_lamb(x_surface, geom)

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def calc_lamb(self, x_surface, geom):
        """Lambertian Reflectance."""
        a = x_surface[self.areal_mix_ind]
        b = x_surface[self.shadow_fraction_ind]
        rfl_n = MultiComponentSurface.calc_lamb(self, x_surface, geom) #subsets with wl idxs 
        rfl = b*a*rfl_n/(1-(1-a)*rfl_n)
        
        return rfl

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to state 
        vector, calculated at x_surface. The dimensions of this are n_bands*n_bands+1"""
        
        # Analytic Solution:
        dlamb = np.zeros((self.n_wl, self.n_wl+1))
        rfl = self.calc_lamb(x_surface, geom)
        a = x_surface[self.areal_mix_ind]
        b = x_surface[self.shadow_fraction_ind]
        #print(b)
        denom = (1-(1-a)*rfl)
        dlamb_drfl = b*a/(denom**2)
        dlamb_da = b*(rfl/denom-a*rfl**2/(denom**2))
        dlamb_db = a*rfl/denom
        dlamb_rfl = np.diag(dlamb_drfl)
        dlamb = np.concatenate([dlamb_rfl, dlamb_da[:,np.newaxis], dlamb_db[:,np.newaxis]], axis=1)
        #dlamb[:, self.areal_mix_ind] = dlamb_da
        #pdb.set_trace()
        return dlamb


    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface."""

        dLs_dsurface = MultiComponentSurface.dLs_dsurface(self, x_surface, geom)

        return dLs_dsurface

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ''
        return 'Component: %i' % self.component(x_surface, geom)
