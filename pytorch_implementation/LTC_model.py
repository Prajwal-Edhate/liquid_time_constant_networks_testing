import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np
import os

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LTCCell(nn.Module):

    # Input size minus one means inputs.shape[-1]

    def __init__(self, num_units, _input_shape_one_minus, num_unfolds = 6, _solver = 'SemiImplicit', _input_mapping = 'Affine'):
        super(LTCCell, self).__init__()
        
        self._num_units = num_units
        self._is_built = False
        self._ode_solver_unfolds = num_unfolds
        if _solver == 'SemiImplicit':
            self._solver = ODESolver.SemiImplicit
        elif _solver == 'Explicit':
            self._solver = ODESolver.Explicit
        else:
            self._solver = ODESolver.RungeKutta

        if _input_mapping == 'Affine':
            self._input_mapping = MappingType.Affine
        elif _input_mapping == 'Linear':
            self._input_mapping = MappingType.Linear
        else:
            self._input_mapping = MappingType.Identity
        self._erev_init_factor = 1
        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1
        self._gleak_init_max = 1
        self._w_min_value = 0.00001
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000
        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None
        self._input_size = _input_shape_one_minus
        self._get_variables()


    def _sigmoid(self, v_pre, mu, sigma):
        
        v_pre = torch.reshape(v_pre, [-1, v_pre.shape[-1], 1])
        mues = v_pre - mu
        x = sigma*mues
        return torch.sigmoid(x)

    # Hybrid euler method
    def _ode_step(self, inputs, state):
        v_pre = state
        #print("I am in ode_step checking first input", inputs.shape)
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        
        for t in range(self._ode_solver_unfolds):
            
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev
            
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            numerator = (self.cm_t * v_pre) + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _ode_step_explicit(self,inputs,state,_ode_solver_unfolds):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation, dim = 1)

        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(_ode_solver_unfolds):
            w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

            w_reduced_synapse = torch.sum(w_activation,dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = torch.sum(sensory_in,dim=1) - v_pre*w_reduced_synapse + torch.sum(synapse_in,dim=1) - v_pre * w_reduced_sensory

            f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        w_reduced_sensory = torch.sum(sensory_w_activation,dim=1)

        # Unfold the mutliply ODE multiple times into one RNN step
        w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

        w_reduced_synapse = torch.sum(w_activation,dim=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = torch.sum(sensory_in,dim=1) - v_pre*w_reduced_synapse + torch.sum(synapse_in,dim=1) - v_pre * w_reduced_sensory
        
        f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):

        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h*self._f_prime(inputs,state)
            k2 = h*self._f_prime(inputs,state+k1*0.5)
            k3 = h*self._f_prime(inputs,state+k2*0.5)
            k4 = h*self._f_prime(inputs,state+k3)

            state = state + 1.0/6*(k1+2*k2+2*k3+k4)

        return state

    def forward(self, inputs, state):
        state = state.to(inputs.device)
        inputs = inputs.to(state.device)
        batch_size = inputs.size(0)
        if not self._is_built:
            # Build the model here        
            self._is_built = True
            

        elif self._input_size != int(inputs.shape[-1]):
            raise ValueError("You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size,
                    int(inputs.shape[-1])
                ))

        inputs = self._map_inputs(inputs)
        output_states = []
        for i in range(batch_size):
            inputs_i = inputs[i]
            if self._solver == ODESolver.Explicit:
                next_state = self._ode_step_explicit(inputs_i, state, self._ode_solver_unfolds)
            elif self._solver == ODESolver.SemiImplicit:
                next_state = self._ode_step(inputs_i, state)
            elif self._solver == ODESolver.RungeKutta:
                next_state = self._ode_step_runge_kutta(inputs_i, state)
            else:
                raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))
            
            output_states.append(next_state)

        next_state = torch.stack(output_states)
        return next_state.view(batch_size,-1)

    def _map_inputs(self, inputs):
        if self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear:
            self.input_w = nn.Parameter(torch.ones_like(inputs))
            self.input_w = self.input_w.to(inputs.device)
            inputs = inputs * self.input_w
        if self._input_mapping == MappingType.Affine:
            self.input_b = nn.Parameter(torch.zeros_like(inputs))
            self.input_b = self.input_b.to(inputs.device)
            inputs = inputs + self.input_b
        return inputs

    def _get_variables(self):
        # Define PyTorch parameters
        self.sensory_mu = nn.Parameter(torch.rand(self._input_size, self._num_units) * 0.5 + 0.3)
        self.sensory_sigma = nn.Parameter(torch.rand(self._input_size, self._num_units) * 5 + 3)
        self.sensory_W = nn.Parameter(torch.rand(self._input_size, self._num_units) * (self._w_init_max - self._w_init_min) + self._w_init_min)
        sensory_erev_init = 2 * (torch.randint(0, 2, (self._input_size, self._num_units)) - 0.5)
        
        self.sensory_erev = nn.Parameter(sensory_erev_init * self._erev_init_factor)

        self.mu = nn.Parameter(torch.rand(self._num_units, self._num_units) * 0.5 + 0.3)
        self.sigma = nn.Parameter(torch.rand(self._num_units, self._num_units) * 5 + 3)
        self.W = nn.Parameter(torch.rand(self._num_units, self._num_units) * (self._w_init_max - self._w_init_min) + self._w_init_min)
        erev_init = 2 * (torch.randint(0, 2, (self._num_units, self._num_units)) - 0.5)
        self.erev = nn.Parameter(erev_init * self._erev_init_factor)

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self._num_units) * 0.4 - 0.2)
        else:
            self.vleak = nn.Parameter(torch.tensor(self._fix_vleak))

        if self._fix_gleak is None:
            gleak_init = torch.rand(self._num_units) * (self._gleak_init_max - self._gleak_init_min) + self._gleak_init_min
            self.gleak = nn.Parameter(gleak_init)
        else:
            self.gleak = nn.Parameter(torch.tensor(self._fix_gleak))

        if self._fix_cm is None:
            cm_init = torch.rand(self._num_units) * (self._cm_init_max - self._cm_init_min) + self._cm_init_min
            self.cm_t = nn.Parameter(cm_init)
        else:
            self.cm_t = nn.Parameter(torch.tensor(self._fix_cm))

    def get_param_constraints(self):
        self.cm_t.data = torch.clamp(self.cm_t.data, self._cm_t_min_value, self._cm_t_max_value)
        cm_clipping_op = self.cm_t
        self.gleak.data = torch.clamp(self.gleak.data, self._gleak_min_value, self._gleak_max_value)
        gleak_clipping_op = self.gleak
        self.W.data = torch.clamp(self.W.data, self._w_min_value, self._w_max_value)
        w_clipping_op = self.W
        self.sensory_W.data = torch.clamp(self.sensory_W.data, self._w_min_value, self._w_max_value)
        sensory_w_clipping_op = self.sensory_W
        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]


    def export_weights(self, dirname, output_weights = None):
        os.makedirs(dirname, exist_ok=True)
        w = self.W.data.numpy()
        erev = self.erev.data.numpy()
        mu = self.mu.data.numpy()
        sigma = self.sigma.data.numpy()
        sensory_w = self.sensory_W.data.numpy()
        sensory_erev = self.sensory_erev.data.numpy()
        sensory_mu = self.sensory_mu.data.numpy()
        sensory_sigma = self.sensory_sigma.data.numpy()
        vleak = self.vleak.data.numpy()
        gleak = self.gleak.data.numpy()
        cm = self.cm_t.data.numpy()

        if output_weights is not None:
            output_w, output_b = output_weights
            np.savetxt(os.path.join(dirname, "output_w.csv"), output_w.data.numpy())
            np.savetxt(os.path.join(dirname, "output_b.csv"), output_b.data.numpy())

        np.savetxt(os.path.join(dirname, "w.csv"), w)
        np.savetxt(os.path.join(dirname, "erev.csv"), erev)
        np.savetxt(os.path.join(dirname, "mu.csv"), mu)
        np.savetxt(os.path.join(dirname, "sigma.csv"), sigma)
        np.savetxt(os.path.join(dirname, "sensory_w.csv"), sensory_w)
        np.savetxt(os.path.join(dirname, "sensory_erev.csv"), sensory_erev)
        np.savetxt(os.path.join(dirname, "sensory_mu.csv"), sensory_mu)
        np.savetxt(os.path.join(dirname, "sensory_sigma.csv"), sensory_sigma)
        np.savetxt(os.path.join(dirname, "vleak.csv"), vleak)
        np.savetxt(os.path.join(dirname, "gleak.csv"), gleak)
        np.savetxt(os.path.join(dirname, "cm.csv"), cm)