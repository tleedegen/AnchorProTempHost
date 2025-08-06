# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 21:35:21 2025

@author: djmiller
"""

import pandas as pd
import numpy as np
from anchor_pro.anchor_pattern_mixin import AnchorPatternMixin

class WoodFastener(AnchorPatternMixin):
    EDGE_DIST_REQS = pd.DataFrame({
        'loading_dir':['perpendicular', 'parallel_compression','parallel_tension', 'parallel_tension'],
        'wood_class':[None, None, 'softwood', 'hardwood'],
        'minimum_for_05': [2, 2, 3.5, 2.5],
        'minimum_for_1': [4, 4, 7 ,5]
    })

    GRAIN_ANGLE = {'X': 0,
                   'Y': np.pi/2}

    def __init__(self, xy_anchors):
        # General Properties
        self.xy_anchors = xy_anchors
        self.anchor_forces = None
        self.Tu_max = None  # Tu_max is not used in calculations, but is referenced by function in the report in order to select the governing anchor object.
        self.max_temp = 100

        
        # Wood Properties (Main Member)
        self.wood_id = None
        self.G = None
        self.E = None
        self.moisture_condition = "Dry"
        self.hardwood_softwood = None  # hardwood or softwood
        self.Bm = None
        self.Dm = None
        self.Am = None
        self.grain_angle = None

        self.theta = None  # Angle w.r.t wood grain
        self.Fe_parallel = None
        self.Fe_perp = None
        self.Fem = None
        self.lm = None

        # "Side Member" Properties (Metal Attachment)
        self.t_steel = None
        self.Fes = None
        self.As = None
        self.g = None  # (Gap between members)
        self.Es = None

        # Fastener Properties
        self.fastener_id = None
        self.fastener_type = None  # "Lag Screw" or "Wood Screw"
        self.D_nom = None
        self.D = None
        self.Fyb = None
        self.length = None

        # Withdrawal Strength Computed Properties
        self.p = None
        self.W = None

        # Lateral Strength Computed Properites
        self.K_theta = None
        self.K_D = None
        self.Rd = None
        self.yield_modes = {}
        self.Z = None

        # Combined Strength Computed Properties
        self.alpha = None
        self.Z_prime = None
        self.W_prime = None
        self.C_M = None
        self.C_t = None
        self.C_g = None
        self.C_delta = None
        self.C_eg = None
        self.C_di = None
        self.C_tn = None
        self.Z_alpha = None

        self.z_alpha_prime = None
        self.V = None
        self.Vy = None
        self.Vx = None
        self.N = None
        self.DCR = None

    def set_member_properties_from_data_table(self, member_data, base_or_wall='base'):
        """ base_or_wall is a key that must be appended to the column parameter
        names to get the correct column names for the equipment data table"""

        # Properties for input tables
        self.G = member_data['G_wood'+'_'+base_or_wall]
        self.E = member_data['E_wood'+'_'+base_or_wall]
        self.moisture_condition = member_data['moisture_condition_wood'+'_'+base_or_wall]
        self.hardwood_softwood = member_data['hardwood_softwood'+'_'+base_or_wall]  # hardwood or softwood
        self.Bm = member_data['Bm'+'_'+base_or_wall]
        self.Dm = member_data['Dm'+'_'+base_or_wall]

        # Computed Properties
        self.grain_angle = WoodFastener.GRAIN_ANGLE[member_data['grain_direction'+'_'+base_or_wall]]
        self.Am = self.Bm * self.Dm


    def set_steel_props(self, t_steel, Fes, g=0):
        # Passed parameters
        self.t_steel = t_steel
        self.Fes = Fes
        self.g = g

        # Inferred parameters
        self.Es = 29000
        self.As = self.Bm * self.t_steel


    def set_fastener_properties(self, fastener_data):
        for key in vars(self).keys():
            if key in fastener_data.keys():
                setattr(self, key, fastener_data.at[key])

        # Calculate penetration values
        self.p = self.length - self.t_steel

    def reference_lateral_design_value(self):
        """ Reference Technical Report 12 for inclusion of gap offset due to gyp board"""
        
        D = self.D
        lm = self.p
        ls = self.t_steel
        Fem = self.Fem
        Fes = self.Fes
        Fyb = self. Fyb # Get this from fastener data
        g = self.g
        
        Rd = self.Rd
        # Rt = lm/ls
        # Re = Fem/Fes

        # k1 = (Re + 2*Re**2*(1 + Rt + Rt**2) + Rt**2*Re**3)**0.5 - Re*(1+Rt) / (1+Re)
        # k2 = -1 + (2 * (1 + Re) + 2 * Fyb * (1 + 2 * Re) * D ** 2 / (3 * Fem * lm ** 2)) ** 0.5
        # k3 = -1 + (2 * (1 + Re) / Re + 2 * Fyb * (2 + Re) * D ** 2 / (3 * Fem * ls ** 2)) ** 0.5

        # Yield Limit Equations (NDS 12.3.1 and TR12 Table 1)
        qs = Fes*D
        qm = Fem*D
        Ms = Fyb * (D**3/6)
        Mm = Fyb * (D**3/6)
        yield_modes_ABCRd = {'II': (1/(4*qs) + 1/(4*qm), ls/2 + g + lm/2, qs*ls**2/4 - qm*lm**2/4, Rd[2]),
                           'IIIm': (1/(2*qs)+1/(4*qm), g + lm/2, -Ms-qm*lm**2/4, Rd[3]),
                           'IIIs': (1/(4*qs) + 1/(2*qm), ls/2 + g, -qs*ls**2/4-Mm, Rd[4]),
                           'IV': (1/(2*qs) + 1/(2*qm), g, -Ms-Mm, Rd[5])}
        
        self.yield_modes['Im'] = D * lm * Fem / Rd[0]
        self.yield_modes['Is'] = D * ls * Fes / Rd[1]
        for mode, (A, B, C, Rd) in yield_modes_ABCRd.items():
            self.yield_modes[mode] = (-B + (B**2 - 4*A*C)**0.5)/(2*A*Rd)
                            # 'II': k1 *  D * ls * Fes / Rd[2],
                            # 'IIIm': k2 * D * lm * Fem / ( (1 + 2 * Re) * Rd[3]),
                            # 'IIIs': k3 * D * ls * Fem / ( (2 + Re) + Rd[4]),
                            # 'IV': D**2 / Rd[5] * (2 * Fem * Fyb / (3 * (1 + Re)))**0.5}

        self.Z = min([val for key, val in self.yield_modes.items()])
        

    def reference_withdrawal_design_value(self):
        if self.fastener_type == 'Lag Screw':
            self.W = 1800*self.G**(3/2) * self.D**(3/4)  # NDS 12.2-1
        elif self.fastener_type == 'Wood Screw':
            self.W = 2850 * self.G**2 * self.D  # NDS 12.2-2
        else:
            raise Exception(f'Wood fastener type "{self.fastener_type}" not supported.')

    def get_loading_dir(self, Vx, Vy):
        """ Determines the direction of loading relative to the wood grain"""
        theta_load = np.atan2(abs(Vy),abs(Vx))
        self.theta = theta_load - self.grain_angle # Angle w.r.t wood grain

        '''COMPUTED PROPERITES'''
        theta_degrees = np.degrees(self.theta)

        # Wood Bearing (NDS 12.3.4 and Table 12.3.3)
        if self.D < 0.25:
            self.Fem = 16600 * self.G ** 1.84
        else:
            self.Fe_parallel = 11200 * self.G
            self.Fe_perp = 6100 * self.G ** 1.45 / (self.D ** 0.5)
            self.Fem = self.Fe_parallel * self.Fe_perp / (self.Fe_parallel * np.sin(self.theta) ** 2 +
                                                          self.Fe_perp * np.sin(self.theta) ** 2)

        # K_theta from table 12.3.1B
        self.K_theta = 1 + 0.25 * (theta_degrees / 90)

        # K_D from Table 12.3.1B
        if self.D <= 0.17:
            self.K_D = 2.2
        elif 0.17 < self.D < 0.25:
            self.K_D = 10 * self.D + 0.5
        else:
            self.K_D = np.nan

        # Reduction term Rd from Table 12.3.1B
        if self.D < 0.25:
            self.Rd = [self.K_D] * 6
        elif (self.D_nom > 0.25) and (self.D < 0.25):  # footnote 1
            self.Rd = [self.K_D * self.K_theta] * 6
        elif 0.25 < self.D < 1:
            self.Rd = [4 * self.K_theta] * 2 + [3.6 * self.K_theta] + [3.2 * self.K_theta] * 3
        else:
            self.Rd = [np.inf] * 6
            raise Exception("Fastener diameter greater than 1\" not permitted")


    def adjustment_factors(self):

        # Wet Service Factor (Table 11.3.3)
        self.C_M = 1.0 if self.moisture_condition == 'Dry' else 0.7  
        
        # Temperature Factor (Table 11.3.4)
        if self.max_temp <= 100:
            self.C_t = 1.0
        else:
            raise Exception('The specified max temperature is not supported')
        
        # Group Action Factor
        if self.D<0.25:
            self.C_g = 1.0
        else:            
            # xy_extents = max(self.xy_anchors.max(axis=0)-self.xy_anchors.min(axis=0))
            self.get_anchor_spacing_matrix()
            As = self.As
            Am = self.Am
            n = len(self.xy_anchors)
            s = np.where(self.spacing_matrix==0,np.inf,self.spacing_matrix).min()
            gamma = 270000 * self.D**1.5
            u = 1 + gamma*(s/2)*(1/(self.E*Am) + 1/(self.Es*As))
            Rea = min(self.Es*As/(self.E*Am), self.E*Am/(self.Es*As))
            m = u - (u**2 - 1)**0.5
            self.C_g = (m * (1 - m**(2*n)) / (n*((1+Rea*m**n)*(1+m)-1+m**(2*n))) ) * (1+Rea)/(1-m)
        
        
        # Geometry Factor 12.5.2
        edge_dist_reqs = {'perpendicular': (2*self.D, 4*self.D),
                          'parallel_compression': (2*self.D, 4*self.D),
                          'parallel_tension_hardwood': (2.5*self.D, 5*self.D),
                          'parallel_tension_softwood': ()}

        if self.D < 0.25:
            self.C_delta = 1.0
        else:
            # Assume fastener is located away from edges
            self.C_delta = 1.0

        # End Grain Factor 12.5.2
        self.C_eg = 1.0  # Assumed no attachment to member end grain
        
        # Diaphragm Factor
        self.C_di = 1.0  # Assumed fasteners are not part of a diaphragm
        
        # Toe Nailed Factor
        self.C_tn = 1.0
        
        self.Kf = 3.32
        self.phi = 0.65
        self.time_factor = 1.0  # Table N3

    def check_fasteners(self):
        Tu_values = np.linalg.norm(self.anchor_forces,axis=2)
        idx_governing_fastener, idx_governing_theta = np.unravel_index(np.argmax(Tu_values),
                                                  self.anchor_forces[:,:,0].shape)
        self.Tu_max = Tu_values[idx_governing_fastener,idx_governing_theta]

        self.reference_withdrawal_design_value()
        self.adjustment_factors()
        self.DCR = 0
        for N, Vx, Vy in self.anchor_forces[idx_governing_fastener,:,:]:
            self.get_loading_dir(Vx, Vy)
            self.reference_lateral_design_value()
            Z_prime = self.Z * self.C_M * self.C_t * self.C_g * self.C_delta * self.C_eg * self.C_di * self.C_tn * self.Kf * self.phi * self.time_factor
            W_prime = self.W * self.C_M * self.C_t * self.C_eg * self.Kf * self.phi * self.time_factor

            shear_demand = (Vx**2 + Vy**2)**0.5
            tension_demand = max(N, 0)
            total_demand = (Vx**2 + Vy**2 + N**2)**0.5
            if np.isclose(total_demand,0):
                cos_alpha = 1
                sin_alpha = 0
            else:
                cos_alpha = shear_demand / total_demand
                sin_alpha = tension_demand / total_demand

            z_alpha_prime = W_prime * self.p * Z_prime / (W_prime*self.p*cos_alpha**2+Z_prime*sin_alpha**2)
            dcr = shear_demand/z_alpha_prime
            if dcr > self.DCR:
                self.N = N
                self.Vx = Vx
                self.Vy = Vy
                self.V = shear_demand
                self.Z_prime = Z_prime
                self.W_prime = W_prime
                self.z_alpha_prime = z_alpha_prime
                self.DCR = dcr
