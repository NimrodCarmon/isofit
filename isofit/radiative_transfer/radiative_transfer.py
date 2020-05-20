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
# Author: Jay E. Fahlen, jay.e.fahlen@jpl.nasa.gov
#

import numpy as np
import logging

from ..core.common import eps
from ..radiative_transfer.modtran import ModtranRT
from ..radiative_transfer.six_s import SixSRT
from ..radiative_transfer.libradtran import LibRadTranRT
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig
from isofit.core.common import load_spectrum

import matplotlib.pyplot as plt
import pdb
class RadiativeTransfer():
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    def __init__(self, full_config: Config):

        # Maintain order when looping for indexing convenience
        config = full_config.forward_model.radiative_transfer

        self.statevec_names = config.statevector.get_element_names()
        self.lut_grid = config.lut_grid

        # TODO: rework this so that we instead initialize an interpolator, that calls
        # RTEs as necessary based on LUT grid or other parameters..which may happen higher up
        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            rte_config: RadiativeTransferEngineConfig = config.radiative_transfer_engines[idx]

            if rte_config.engine_name == 'modtran':
                rte = ModtranRT(rte_config, full_config)
            elif rte_config.engine_name == 'libradtran':
                rte = LibRadTranRT(rte_config, full_config)
            elif rte_config.engine_name == '6s':
                rte = SixSRT(rte_config, full_config)
            else:
                # Should never get here, checked in config
                raise AttributeError(
                    'Invalid radiative transfer engine name: {}'.format(rte_config.engine_name))

            self.rt_engines.append(rte)

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        self.wl = np.concatenate([RT.wl for RT in self.rt_engines])

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = np.concatenate([RT.solar_irr for RT in self.rt_engines])
        # These should all be the same so just grab one
        self.coszen = [RT.coszen for RT in self.rt_engines][0]

    def xa(self):
        """Pull the priors from each of the individual RTs.
        """
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs.
        """
        return np.diagflat(np.power(np.array(self.prior_sigma), 2))

    def get(self, x_RT, geom):

        ret = []
        for RT in self.rt_engines:
            ret.append(RT.get(x_RT, geom))

        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        '''
        L_atm is the path radiance. In the VSWIR case, it is calculated by running modtran with zero reflectance,
        and taking the "TOTAL_RAD" output, divided by the TOA solar irradiance. The TIR case is thermal upwelling.

        L_down_transmittence is the product of the global downwelling flux and the total (direct+diffuse) transmittence.
        The global flux is calculated by adjusting the TOA Solar Irrad product to radiance units (cos * pi-1)
        The global tranmittence is the sum of the A and B coefficients calculated by modtran, convolved and written to .chn.

        '''
        #pdb.set_trace()
        r = self.get(x_RT, geom)
        L_atm = self.get_L_atm(x_RT, geom)
        L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

        L_up = self.get_L_up(x_RT, geom)
        L_up = L_up + Ls * r['transup']
        '''
        ret = L_atm + \
            L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl) + \
            L_up

        '''
        #trans_dir = self.get_transm_dir
        #trans_dif = self.get_trans_dif
        I = self.get_Solar_Illumination(x_RT, geom)
        bck_rfl = self.get_background_reflectance()
        neigh_rfl = self.get_neighbor_reflectance()
        #nbr_rfl =
        #bck_rfl = rfl
        #neigh_rfl = rfl
        #pdb.set_trace()
        ret = L_atm + \
            I / (1.0-r['sphalb'] * bck_rfl) * neigh_rfl * r['transm_dif'] + \
            I / (1.0-r['sphalb'] * bck_rfl) * 0.86 * rfl * r['transm_dir'] + \
            L_up
        #pdb.set_trace()

        return ret



    def get_Solar_Illumination(self, x_RT, geom):
        Illum = []
        for RT in self.rt_engines:
            Illum.append(RT.get_illumination(x_RT, geom))
        return np.hstack(Illum)

    def get_neighbor_reflectance(self):
        file2use = 'outputs/Backman_neighbors_ref.txt'
        ref = np.asarray(load_spectrum(file2use))
        ref = ref[0,:]
        return ref

    def get_background_reflectance(self):
        file2use = 'outputs/background_ref.txt'
        ref = np.asarray(load_spectrum(file2use))
        ref = ref[0,:]
        return ref


    def get_L_atm(self, x_RT, geom):
        L_atms = []
        for RT in self.rt_engines:
            L_atms.append(RT.get_L_atm(x_RT, geom))
        return np.hstack(L_atms)

    def get_L_down_transmitted(self, x_RT, geom):
        L_downs = []
        for RT in self.rt_engines:
            L_downs.append(RT.get_L_down_transmitted(x_RT, geom))
        return np.hstack(L_downs)

    def get_L_up(self, x_RT, geom):
        '''L_up is provided by the surface model, so just return
        0 here. The commented out code here is for future updates.'''
        #L_ups = []
        # for key, RT in self.RTs.items():
        #    L_ups.append(RT.get_L_up(x_RT, geom))
        # return s.hstack(L_ups)

        return 0.

    def drdn_dRT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls,
                 dLs_dsurface, geom):

        # first the rdn at the current state vector
        rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + np.eye(len(x_RT))*eps
        for x_RT_perturb in list(x_RTs_perturb):
            rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
            K_RT.append((rdne-rdn) / eps)
        K_RT = np.array(K_RT).T

        # Get K_surface
        r = self.get(x_RT, geom)
        L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

        # The reflected downwelling light is:
        # L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl), or
        # L_down_transmitted * rho_scaled_for_multiscattering
        # This term is the derivative of rho_scaled_for_multiscattering
        drho_scaled_for_multiscattering_drfl = 1. / (1 - r['sphalb']*rfl)**2

        drdn_drfl = L_down_transmitted * drho_scaled_for_multiscattering_drfl
        drdn_dLs = r['transup']
        K_surface = drdn_drfl[:, np.newaxis] * drfl_dsurface + \
            drdn_dLs[:, np.newaxis] * dLs_dsurface

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):

        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))

        else:
            # first the radiance at the current state vector
            r = self.get(x_RT, geom)
            rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

            # unknown parameters modeled as random variables per
            # Rodgers et al (2000) K_b matrix.  We calculate these derivatives
            # by finite differences
            Kb_RT = []
            perturb = (1.0+eps)
            for unknown in self.bvec:
                if unknown == 'H2O_ABSCO' and 'H2OSTR' in self.statevec_names:
                    i = self.statevec_names.index('H2OSTR')
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
                    Kb_RT.append((rdne-rdn) / eps)

        Kb_RT = np.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.rt_engines:
            ret.append(RT.summarize(x_RT, geom))
        ret = '\n'.join(ret)
        return ret

    def pack_arrays(self, list_of_r_dicts):
        """Take the list of dict outputs from each RTM (in order of RTs) and
        stack their internal arrays in the same order.
        """
        r_stacked = {}
        for key in list_of_r_dicts[0].keys():
            temp = [x[key] for x in list_of_r_dicts]
            r_stacked[key] = np.hstack(temp)
        return r_stacked
