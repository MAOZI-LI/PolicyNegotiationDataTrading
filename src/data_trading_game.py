import math
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import copy
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['savefig.transparent'] = False
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams.update({
    'font.size': 14,            
    'axes.labelsize': 18,      
    'axes.labelweight': 'bold', 
    'axes.titlesize': 18,      
    'xtick.labelsize': 14,      
    'ytick.labelsize': 14,     
    'legend.fontsize': 12,      
    'font.weight': 'bold',      
})
import tqdm
# from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec


def compute_quality(t, qmax, t0, decay_type='linear', eta=0.1, alpha=2.0):
    """
    Data quality as a function of physical delivery time t.
    For exponential decay, eta is dimensional and q=qmax*exp(-eta*t).
    For linear/power decay, t0 is the obsolescence time; q is truncated to be non-negative.
    """
    t = max(0.0, float(t))
    t0 = max(float(t0), 1e-12)
    ratio = min(max(t / t0, 0.0), 1.0)
    if decay_type == 'linear':
        q = qmax * (1.0 - ratio)
    elif decay_type == 'exp':
        q = qmax * math.exp(-eta * t)
    elif decay_type == 'power':
        if alpha is None or float(alpha) <= 0:
            raise ValueError('power decay requires alpha > 0')
        alpha = float(alpha)
        q = qmax * (1.0 - ratio ** alpha)
    else:
        raise ValueError(f'unknown decay_type: {decay_type}')
    return max(q, 0.0)


class Market:
    def __init__(self, seed, data_size, lamda, q_0, gamma, t1_t0, accuracy, tao,
                 decay_type='linear', decay_eta=0.1, decay_alpha=2.0, t0_abs=1.0,
                 delivery_time_type='log', delivery_beta=1.5,
                 delivery_theta1=0.8, delivery_theta2=0.5,
                 delivery_piece_c1_scale=0.5, delivery_piece_c2_scale=1.5,
                 risk_cost_type='quadratic', risk_power=2.0,
                 risk_threshold=0.5, risk_threshold_penalty=4.0):
        self.cacheEnabled = False
        self.seed = seed
        self.data_size = data_size
        # unit risk cost
        self.lamda = lamda
        self.q_0 = q_0
        self.gamma = gamma
        self.t1_t0 = t1_t0
        self.t0_abs = float(t0_abs)
        self.t1_abs = self.t1_t0 * self.t0_abs
        self.accuracy = accuracy
        self.tao = tao
        # data quality decay: linear (default), exp, power
        self.decay_type = decay_type
        self.decay_eta = decay_eta
        self.decay_alpha = decay_alpha
        # delivery-time function: log (main model) or power robustness alternative
        self.delivery_time_type = delivery_time_type
        if delivery_beta is None or float(delivery_beta) <= 0:
            raise ValueError('power delivery-time function requires beta > 0')
        self.delivery_beta = float(delivery_beta)
        self.delivery_theta1 = float(delivery_theta1)
        self.delivery_theta2 = float(delivery_theta2)
        if not (0.0 <= self.delivery_theta2 < self.delivery_theta1 <= 1.0):
            raise ValueError('piecewise delivery-time function requires 0 <= theta2 < theta1 <= 1')
        self.delivery_piece_c1_scale = float(delivery_piece_c1_scale)
        self.delivery_piece_c2_scale = float(delivery_piece_c2_scale)
        if self.delivery_piece_c1_scale < 0 or self.delivery_piece_c2_scale < 0:
            raise ValueError('piecewise delivery-time slopes must be non-negative')
        # Provider risk-cost function. The main model uses quadratic cost.
        # Alternative forms are solved numerically because Stage 3 changes.
        self.risk_cost_type = risk_cost_type
        self.risk_power = float(risk_power)
        if self.risk_cost_type == 'cubic':
            self.risk_power = 3.0
        self.risk_threshold = float(risk_threshold)
        self.risk_threshold_penalty = float(risk_threshold_penalty)
        if self.risk_cost_type == 'power' and self.risk_power <= 0:
            raise ValueError('power risk-cost function requires risk_power > 0')
        if self.risk_cost_type == 'threshold':
            self.risk_threshold = min(max(self.risk_threshold, 0.0), 1.0)
        # unit product price
        self.p1 = 0
        # attribute name -> attribute
        self.attributes = {}

    def add_attribute(self, name, type, w, rho, L_size, B_size, L_join_B_size):
        if type not in ['continuous', 'discrete']:
            raise Exception(f'type {type} should be continuous or discrete')
        if name in self.attributes:
            raise Exception(f'attribute {name} already exists')
        if L_join_B_size > L_size:
            raise Exception(f'L_join_B_size {L_join_B_size} > L_size {L_size}')
        if L_join_B_size > B_size:
            raise Exception(f'L_join_B_size {L_join_B_size} > B_size {B_size}')
        self.attributes[name] = {
            'type': type,
            'w': w,
            'rho': rho,
            'L_size': L_size,
            'B_size': B_size,
            'L_join_B_size': L_join_B_size,
            'policy_set': False,
            'x': 0,
            'y': 0,
            'z': 0,
            'p2': -1
        }

    def attrs_for_test(self):
        continuous_attr_name = None
        discrete_attr_name = None
        for attr_name in self.attributes:
            type = self.attributes[attr_name]['type']
            if continuous_attr_name is None and type == 'continuous':
                continuous_attr_name = attr_name
            if discrete_attr_name is None and type == 'discrete':
                discrete_attr_name = attr_name
            if continuous_attr_name is not None and discrete_attr_name is not None:
                break
        attrs = []
        if continuous_attr_name is not None:
            attrs.append(continuous_attr_name)
        if discrete_attr_name is not None:
            attrs.append(discrete_attr_name)
        # attrs = ['A1', 'A6']
        return attrs

    def weight_regularize(self, w_only=False, rho_only=False):
        w_sum = 0
        rho_sum = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            w_sum += attr['w']
            rho_sum += attr['rho']
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if not rho_only:
                attr['w'] = attr['w'] / w_sum
            if not w_only:
                attr['rho'] = attr['rho'] / rho_sum

    def set_policy(self, name, x, y, z):
        """
        :param name: attribute name
        :param x: |T join (L sub B)|, 0 <= x <= |L| - |L join B|
        :param y: |T join (B sub L)|, 0 <= y <= |B| - |L join B|
        :param z: |T join (L join B)|, 0 <= z <= |L join B|
        :return:
        """
        if name not in self.attributes:
            raise Exception(f'attribute {name} not found')
        attr = self.attributes[name]
        if attr['type'] == 'discrete':
            if type(x) != int or type(y) != int or type(z) != int:
                raise Exception(f'x, y, z should be integer')
        delta = 1E-5
        if x > attr['L_size'] - attr['L_join_B_size'] + delta:
            raise Exception(f'x {x} > L_size {attr["L_size"]} - L_join_B_size {attr["L_join_B_size"]}')
        if y > attr['B_size'] - attr['L_join_B_size'] + delta:
            raise Exception(f'y {y} > B_size {attr["B_size"]} - L_join_B_size {attr["L_join_B_size"]}')
        if z > attr['L_join_B_size'] + delta:
            raise Exception(f'z {z} > L_join_B_size {attr["L_join_B_size"]}')
        attr['x'] = x
        attr['y'] = y
        attr['z'] = z
        attr['policy_set'] = True

    def set_p2(self, name, p2):
        if name not in self.attributes:
            raise Exception(f'attribute {name} not found')
        if p2 < 0:
            raise Exception(f'price p2 should be non-negative')
        attr = self.attributes[name]
        attr['p2'] = p2

    def variable_range(self, variable, attr_name):
        if attr_name not in self.attributes:
            raise Exception(f'attribute {attr_name} not found')
        attr = self.attributes[attr_name]
        if variable == 'x':
            return 0, attr['L_size'] - attr['L_join_B_size']
        if variable == 'y':
            return 0, attr['B_size'] - attr['L_join_B_size']
        if variable == 'z':
            return 0, attr['L_join_B_size']
        else:
            raise Exception(f'variable should be x, y or z')

    def point_sample(self, variable, attr_name, n_points):
        min, max = self.variable_range(variable, attr_name)
        step = (max - min) / n_points
        return [min + i * step for i in range(n_points)] + [max]

    def broker_satisfiability(self):
        satisfiability = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if not attr['policy_set']:
                raise Exception(f'policy not set for attribute {attr["name"]}')
            satisfiability += attr['w'] * (attr['x'] + attr['z']) / attr['L_size']
        return satisfiability

    def delivery_delta_t(self):
        """Normalized delivery-time factor t/t_1 under the selected delivery function."""
        if self.delivery_time_type == 'log':
            delta_t = 1
            for attr_name in self.attributes:
                attr = self.attributes[attr_name]
                if not attr['policy_set']:
                    raise Exception(f'policy not set for attribute {attr_name}')
                if attr['x'] + attr['z'] == 0:
                    return 1
                psr_i = (attr['x'] + attr['z']) / attr['L_size']
                delta_t -= attr['w'] * math.log(psr_i, math.e)
            return delta_t
        if self.delivery_time_type in ('power', 'nonlinear'):
            delta_t = 1
            for attr_name in self.attributes:
                attr = self.attributes[attr_name]
                if not attr['policy_set']:
                    raise Exception(f'policy not set for attribute {attr_name}')
                if attr['x'] + attr['z'] == 0:
                    psr_i = 0.0
                else:
                    psr_i = (attr['x'] + attr['z']) / attr['L_size']
                psr_i = min(max(psr_i, 0.0), 1.0)
                delta_t += attr['w'] * math.pow(1.0 - psr_i, self.delivery_beta)
            return delta_t
        if self.delivery_time_type in ('piecewise', 'threshold'):
            psr = self.broker_satisfiability()
            psr = min(max(psr, 0.0), 1.0)
            if psr >= self.delivery_theta1:
                return 1.0
            if psr >= self.delivery_theta2:
                return 1.0 + self.delivery_piece_c1_scale * (self.delivery_theta1 - psr)
            return (
                1.0
                + self.delivery_piece_c1_scale * (self.delivery_theta1 - self.delivery_theta2)
                + self.delivery_piece_c2_scale * (self.delivery_theta2 - psr)
            )
        raise ValueError(f'unknown delivery_time_type: {self.delivery_time_type}')

    def time_ratio(self):
        """Dimensionless delivery-time ratio t/t_0."""
        return self.delivery_time() / max(self.t0_abs, 1e-12)

    def delivery_time(self):
        """Physical delivery time t=t_1*delivery_delta_t()."""
        self.t1_abs = self.t1_t0 * self.t0_abs
        return self.t1_abs * self.delivery_delta_t()

    def quality_decay_multiplier(self, time_ratio):
        """Map t/t_0 to q/q_max; exp uses the corresponding physical time."""
        t = max(0.0, time_ratio) * self.t0_abs
        return compute_quality(
            t, 1.0, self.t0_abs,
            decay_type=self.decay_type, eta=self.decay_eta, alpha=self.decay_alpha,
        ) / 1.0

    def data_quality(self):
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if not attr['policy_set']:
                raise Exception(f'policy not set for attribute {attr_name}')
            if attr['p2'] < 0:
                raise Exception(f'price p2 not set for attribute {attr_name}')
            if attr['x'] + attr['z'] == 0:
                return 0
        return compute_quality(
            self.delivery_time(), self.q_0, self.t0_abs,
            decay_type=self.decay_type, eta=self.decay_eta, alpha=self.decay_alpha,
        )

    def seller_risk_cost(self):
        return self.seller_cost()

    def seller_revenue(self):
        revenue = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if not attr['policy_set']:
                raise Exception(f'policy not set for attribute {attr["name"]}')
            if attr['p2'] < 0:
                raise Exception(f'price p2 not set for attribute {attr["name"]}')
            revenue += attr['p2'] * (attr['x'] + attr['z']) / attr['L_size']
        return revenue * self.data_size

    def seller_cost(self):
        risk = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if not attr['policy_set']:
                raise Exception(f'policy not set for attribute {attr["name"]}')
            risk += attr['rho'] * self.risk_cost_multiplier((attr['y'] + attr['z']) / attr['B_size'])
        return self.lamda * self.data_size * risk

    def risk_cost_multiplier(self, exposure_ratio):
        """Dimensionless risk cost f(r), where r=(y+z)/|B|.

        quadratic is the benchmark f(r)=r^2. The threshold variant keeps the
        benchmark shape below the exposure threshold and adds an extra convex
        penalty after the threshold to represent compliance/security escalation.
        """
        r = min(max(float(exposure_ratio), 0.0), 1.0)
        if self.risk_cost_type == 'quadratic':
            return r * r
        if self.risk_cost_type == 'linear':
            return r
        if self.risk_cost_type == 'cubic':
            return r ** 3
        if self.risk_cost_type == 'power':
            return r ** self.risk_power
        if self.risk_cost_type == 'threshold':
            excess = max(0.0, r - self.risk_threshold)
            return r * r + self.risk_threshold_penalty * excess * excess
        raise ValueError(f'unknown risk_cost_type: {self.risk_cost_type}')

    def seller_utility(self):
        return self.seller_revenue() - self.seller_cost()

    def broker_revenue(self):
        return self.p1 * self.data_quality()

    def broker_cost(self):
        return self.seller_revenue() * 0.8 + self.tao * self.data_size * self.accuracy * self.accuracy

    def broker_utility(self):
        return self.broker_revenue() - self.broker_cost()

    def buyer_revenue(self):
        return self.gamma * math.log(1 + self.data_quality(), math.e)

    def buyer_cost(self):
        return self.broker_revenue()

    def buyer_utility(self):
        return self.buyer_revenue() - self.buyer_cost()

    def build_computation_cache(self):
        a_base = self.q_0 * self.t1_t0 / (2 * self.lamda * self.data_size)
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            attr['cache:p2/z'] = 2 * self.lamda * attr['rho'] * attr['L_size'] / (attr['B_size'] * attr['B_size'])
            attr['cache:a/p1'] = a_base * attr['w'] * math.pow(attr['B_size'], 2) / attr['rho']
            if attr['type'] == 'discrete':
                if attr['L_join_B_size'] == attr['L_size']:
                    attr['cache:ub0/p1'] = -float('inf')
                else:
                    attr['cache:ub0/p1'] = self.q_0 * self.t1_t0 * attr['w'] * math.log((attr['L_size'] - attr['L_join_B_size']) / attr['L_size'], math.e)
        self.cacheEnabled = True

    def clean_computation_cache(self):
        self.cacheEnabled = False
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            # delete all keys starting with 'cache:'
            keys = list(attr.keys())
            for key in keys:
                if key.startswith('cache:'):
                    del attr[key]

    def seller_best_strategy(self):
        if self.risk_cost_type != 'quadratic':
            return self.seller_best_strategy_numerical()
        if not self.cacheEnabled:
            self.build_computation_cache()
        best_strategy = {}
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            x_best = attr['L_size'] - attr['L_join_B_size']
            y_best = 0
            # z* = p2 * |B|^2 / (2 * lambda * rho * |L|)
            z_best = attr['p2'] / attr['cache:p2/z']
            if attr['type'] == 'discrete':
                # Stage-3 provider best response is independent of q(t). Keep the
                # original paper's rounding/tie rule: fractional part >= 0.5 rounds up.
                if z_best - math.floor(z_best) >= 0.5:
                    z_best = math.floor(z_best) + 1
                else:
                    z_best = math.floor(z_best)
            best_strategy[attr_name] = {
                'x': x_best,
                'y': y_best,
                'z': z_best if (z_best <= attr['L_join_B_size']) else attr['L_join_B_size']
            }
        return best_strategy

    def seller_best_strategy_numerical(self):
        """Stage-3 provider best response for non-quadratic risk cost.

        Provider utility is separable by attribute after fixing p_T. For each
        attribute, x is set to its maximum because it increases payment without
        increasing exposure, and y is set to zero because it only increases risk.
        We then maximize over z in [0, |V_T cap R|]. Discrete attributes enumerate
        every feasible integer z, preserving deterministic tie-breaking by
        choosing the larger z when utilities are numerically tied; continuous
        attributes use deterministic grid refinement.
        """
        best_strategy = {}
        for attr_name, attr in self.attributes.items():
            x_best = attr['L_size'] - attr['L_join_B_size']
            y_best = 0
            z_hi = max(float(attr['L_join_B_size']), 0.0)

            def util_z(z):
                psr_i = (x_best + z) / max(float(attr['L_size']), 1e-12)
                exposure = z / max(float(attr['B_size']), 1e-12)
                return self.data_size * (
                    attr['p2'] * psr_i
                    - self.lamda * attr['rho'] * self.risk_cost_multiplier(exposure)
                )

            if attr['type'] == 'discrete':
                candidates = range(int(attr['L_join_B_size']) + 1)
                best_z, best_u = 0, -float('inf')
                for z in candidates:
                    u = util_z(z)
                    if u > best_u + 1e-12 or abs(u - best_u) <= 1e-12 and z > best_z:
                        best_z, best_u = z, u
                z_best = best_z
            else:
                if z_hi <= 0:
                    z_best = 0.0
                else:
                    candidates = [0.0, z_hi]
                    denom = max(self.lamda * attr['rho'] * attr['L_size'], 1e-12)
                    if self.risk_cost_type == 'linear':
                        # Linear risk has a constant marginal cost, so the
                        # optimum is one of the interval endpoints.
                        pass
                    elif self.risk_cost_type == 'cubic':
                        z = math.sqrt(max(attr['p2'] * attr['B_size'] ** 3 / (3.0 * denom), 0.0))
                        candidates.append(z)
                    elif self.risk_cost_type == 'power':
                        if abs(self.risk_power - 1.0) > 1e-12:
                            z = (
                                attr['p2'] * attr['B_size'] ** self.risk_power
                                / max(self.lamda * attr['rho'] * attr['L_size'] * self.risk_power, 1e-12)
                            ) ** (1.0 / (self.risk_power - 1.0))
                            candidates.append(z)
                    elif self.risk_cost_type == 'threshold':
                        theta_z = self.risk_threshold * attr['B_size']
                        candidates.append(theta_z)
                        z_low = attr['p2'] * attr['B_size'] ** 2 / (2.0 * denom)
                        candidates.append(z_low)
                        r_high = (
                            attr['p2'] * attr['B_size'] / (2.0 * denom)
                            + self.risk_threshold_penalty * self.risk_threshold
                        ) / (1.0 + self.risk_threshold_penalty)
                        candidates.append(r_high * attr['B_size'])
                    else:
                        # Conservative fallback for future custom risk forms.
                        candidates.extend(np.linspace(0.0, z_hi, 41))
                    z_best, best_u = 0.0, util_z(0.0)
                    for z in candidates:
                        z = min(max(float(z), 0.0), z_hi)
                        u = util_z(z)
                        if u > best_u + 1e-12 or abs(u - best_u) <= 1e-12 and z > z_best:
                            z_best, best_u = z, u
            best_strategy[attr_name] = {'x': x_best, 'y': y_best, 'z': z_best}
        return best_strategy

    def broker_utility_without_policy(self, p2s, z2s):
        delta_t = 1
        cost = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            s = (attr['L_size'] - attr['L_join_B_size'] + z2s[attr_name]) / attr['L_size']
            delta_t = delta_t - attr['w'] * math.log(s, math.e)
            cost += p2s[attr_name] * s
        time_ratio = self.t1_t0 * delta_t
        q = compute_quality(time_ratio * self.t0_abs, self.q_0, self.t0_abs,
                            decay_type=self.decay_type, eta=self.decay_eta,
                            alpha=self.decay_alpha)
        return self.p1 * q - cost * self.data_size - self.tao * self.data_size * self.accuracy * self.accuracy

    def broker_best_strategy(self):
        if not self.cacheEnabled:
            self.build_computation_cache()
        best_strategy = {}
        best_zs = {}
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            # a = a_base * w * |B|^2 / rho
            a = attr['cache:a/p1'] * self.p1
            # x = |L| - |L join B|
            x = attr['L_size'] - attr['L_join_B_size']
            # print(f'attr: {attr_name}, a: {a}, x: {x}')
            if attr['type'] == 'discrete':
                best_z = (math.sqrt((x + 0.5) * (x + 0.5) + 8 * a) - 3 * x + 0.5) / 4
                if best_z < 1:
                    s = (x + 1) / attr['L_size']
                    ub1 = self.p1 * self.q_0 * self.t1_t0 * attr['w'] * math.log(s, math.e) - 0.5 * attr['cache:p2/z'] * s * self.data_size
                    best_z = 0 if ub1 <= attr['cache:ub0/p1'] * self.p1 else 1
                elif best_z > attr['L_join_B_size']:
                    best_z = attr['L_join_B_size']
                else:
                    left = math.floor(best_z)
                    s_left = (x + left) / attr['L_size']
                    s_right = (x + left + 1) / attr['L_size']
                    base = self.p1 * self.q_0 * self.t1_t0 * attr['w']
                    ub_left = base * math.log(s_left, math.e) - (left - 0.5) * attr['cache:p2/z'] * s_left * self.data_size
                    ub_right = base * math.log(s_right, math.e) - (left + 0.5) * attr['cache:p2/z'] * s_right * self.data_size
                    best_z = left if ub_left >= ub_right else left + 1
                best_zs[attr_name] = best_z
                best_strategy[attr_name] = attr['cache:p2/z'] * max(best_z - 0.5, 0)
            else:
                best_z = (math.sqrt(x * x + 8 * a) - 3 * x) / 4
                if best_z > attr['L_join_B_size']:
                    best_z = attr['L_join_B_size']
                elif best_z < 0:
                    best_z = 0
                best_zs[attr_name] = best_z
                best_strategy[attr_name] = attr['cache:p2/z'] * best_z
        #if self.broker_utility_without_policy(best_strategy, best_zs) <= 0:
         #   return None, None
        return best_strategy, best_zs

    def buyer_utility_derivative(self, p1, attrs_left, attrs_middle, discrete_attrs_z, splitter=None):
        if not self.cacheEnabled:
            self.build_computation_cache()
        self.p1 = p1
        part1 = 0
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            x = attr['L_size'] - attr['L_join_B_size']
            if attr['type'] == 'continuous':
                if attr_name in attrs_left or (splitter is not None and splitter[1] == 's' and splitter[2] == attr_name):
                    self.set_p2(attr_name, 0)
                    self.set_policy(attr_name, x, 0, 0)
                    continue
                if attr_name not in attrs_middle or (splitter is not None and splitter[1] == 'e' and splitter[2] == attr_name):
                    z = attr['L_join_B_size']
                else:
                    a = attr['cache:a/p1'] * self.p1
                    sqrt = math.sqrt(x * x + 8 * a)
                    z = (sqrt - 3 * x) / 4
                    if x + z == 0:
                        part1 = float('inf')
                    else:
                        part1 += attr['w'] * attr['cache:a/p1'] / (x + z) / sqrt
                p2 = z * attr['cache:p2/z']
            else:
                z = discrete_attrs_z[attr_name]
                p2 = (z - 0.5) * attr['cache:p2/z'] if z != 0 else 0
            self.set_p2(attr_name, p2)
            self.set_policy(attr_name, x, 0, z)
        data_quality = self.data_quality()
        part2 = self.q_0 * self.t1_t0 * (self.gamma / (1 + data_quality) - self.p1)
        return part1 * part2 - data_quality, part2 <= 0, data_quality, self.broker_utility()

    def buyer_best_strategy(self):
        if not self.cacheEnabled:
            self.build_computation_cache()
        co = self.data_size / self.q_0 / self.t1_t0
        splitters = []
        intervals = {}
        for attr_name in self.attributes:
            attr = self.attributes[attr_name]
            x = attr['L_size'] - attr['L_join_B_size']
            if attr['type'] == 'discrete':
                co_i = co * attr['cache:p2/z'] / attr['w'] / attr['L_size']
                for point in range(attr['L_join_B_size']):
                    if point == 0:
                        if x == 0:
                            continue
                        splitter = 0.5 * (x + 1) * co_i / math.log((x + 1) / x, math.e)
                    else:
                        splitter = (2 * point + x + 0.5) * co_i / math.log((x + 1 + point) / (x + point), math.e)
                    splitters.append((splitter, point + 1, attr_name))
            else:
                start = x * x / attr['cache:a/p1']
                end = attr['L_size'] * (attr['L_size'] + attr['L_join_B_size']) / attr['cache:a/p1']
                splitters.append((start, 's', attr_name))
                splitters.append((end, 'e', attr_name))
                intervals[attr_name] = (start, end)
        #print(f'intervals: {intervals}')
        splitters = sorted(splitters, key=lambda t: t[0])
        attrs_middle = []
        attrs_left = []
        discrete_attrs_z = {}
        for attr_name in self.attributes:
            if self.attributes[attr_name]['type'] == 'continuous':
                attrs_left.append(attr_name)
            else:
                discrete_attrs_z[attr_name] = 0
        best_p1 = set()
        best_p1.add(0)
        best_discrete_attrs_z = discrete_attrs_z.copy()
        max_buyer_utility = 0
        for i in range(len(splitters)):
            segment_left = splitters[i - 1][0] if i > 0 else 0
            segment_right = splitters[i][0]
            splitter = splitters[i-1] if splitters[i-1][1] == 's' else None
            derivative_left, _, data_quality_left, broker_utility_left = self.buyer_utility_derivative(segment_left, attrs_left, attrs_middle, discrete_attrs_z, splitter)
            splitter = splitters[i] if splitters[i][1] == 'e' else None
            derivative_right, flag, data_quality_right, broker_utility_right = self.buyer_utility_derivative(segment_right, attrs_left, attrs_middle, discrete_attrs_z, splitter)
            #print(f'p1: [{segment_left}, {segment_right}], attrs_left: {attrs_left}, attrs_middle: {attrs_middle}')
            #print(f'derivative_left: {derivative_left}, derivative_right: {derivative_right}')
            if broker_utility_right > 0:
                if broker_utility_left <= 0:
                    broker_zero_left = segment_left
                    broker_zero_right = segment_right
                    while broker_zero_right - broker_zero_left > 0.000001:
                        p1 = (broker_zero_left + broker_zero_right) / 2
                        _, _, _, broker_utility = self.buyer_utility_derivative(p1, attrs_left, attrs_middle, discrete_attrs_z)
                        if broker_utility <= 0:
                            broker_zero_left = p1
                        else:
                            broker_zero_right = p1
                    segment_left = broker_zero_right
                    derivative_left, _, data_quality_left, _ = self.buyer_utility_derivative(
                        segment_left, attrs_left, attrs_middle, discrete_attrs_z)
                if derivative_left <= 0:
                    p1 = segment_left
                    data_quality = data_quality_left
                elif derivative_right >= 0:
                    p1 = segment_right
                    data_quality = data_quality_right
                else:
                    while segment_right - segment_left > 0.000001:
                        p1 = (segment_right + segment_left) / 2
                        derivative_p1, _, data_quality, _ = self.buyer_utility_derivative(p1, attrs_left, attrs_middle, discrete_attrs_z)
                        if derivative_p1 > 0:
                            segment_left = p1
                        else:
                            segment_right = p1
                    p1 = (segment_right + segment_left) / 2
                    _, _, data_quality, _ = self.buyer_utility_derivative(p1, attrs_left, attrs_middle, discrete_attrs_z)
                buyer_utility = self.gamma * math.log(1 + data_quality, math.e) - p1 * data_quality
                if buyer_utility > max_buyer_utility:
                    max_buyer_utility = buyer_utility
                    best_discrete_attrs_z = discrete_attrs_z.copy()
                    best_p1.clear()
                    best_p1.add(p1)
                elif buyer_utility == max_buyer_utility:
                    best_p1.add(p1)
            if flag:
                break
            splitter = splitters[i]
            if splitter[1] == 's':
                attrs_left.remove(splitter[2])
                attrs_middle.append(splitter[2])
            elif splitter[1] == 'e':
                attrs_middle.remove(splitter[2])
            else:
                discrete_attrs_z[splitter[2]] = splitter[1]
        return splitters, best_p1, max_buyer_utility, best_discrete_attrs_z


def increasing_array(array):
    # check if the array is increasing
    for i in range(1, len(array)):
        if array[i] < array[i-1]:
            return False
    return True


def getUpperInt(x):
    d = int(math.log(x, 10))
    flag = True
    while True:
        ret = (math.floor(x / pow(10, d)) + (5 if flag else 1)) * pow(10, d)
        delta = (ret - x) / x
        if delta < 1/9:
            return ret
        flag = not flag
        if flag:
            d -= 1

def getLowerInt(x):
    if x < 0:
        return -getUpperInt(-x)
    if x == 0:
        return 0
    d = int(math.log(x, 10))
    while True:
        ret = (math.ceil(x / pow(10, d)) - 1)* pow(10, d)
        delta = (x - ret) / x
        if delta < 1/9:
            return ret
        d -= 1


def draw_utility_figure(x_label, x_array, utility_arrays_list, nash_point, scales, xlims=None, ylims=None, subfigs=False):
    buyer_utility_array, broker_utility_array, seller_utility_array = utility_arrays_list[0]
    max_seller_utility = getUpperInt(max(seller_utility_array))
    min_seller_utility = max(0, getLowerInt(min(seller_utility_array)))  # 确保最小值为0
    max_broker_utility = getUpperInt(max(broker_utility_array))
    min_broker_utility = max(0, getLowerInt(min(broker_utility_array)))  # 确保最小值为0
    max_buyer_utility = getUpperInt(max(buyer_utility_array))
    min_buyer_utility = max(0, getLowerInt(min(buyer_utility_array)))  # 确保最小值为0
    print(f'max_seller_utility: {max_seller_utility}, min_seller_utility: {min_seller_utility}')
    print(f'max_broker_utility: {max_broker_utility}, min_broker_utility: {min_broker_utility}')
    print(f'max_buyer_utility: {max_buyer_utility}, min_buyer_utility: {min_buyer_utility}')

    fig = plt.figure()
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax = fig.add_subplot(gs[0:3, 0:1])    # The big subplot
    fig.subplots_adjust(wspace=0.3, hspace=0.02)

    for _ax in [ax1, ax2, ax3]:
        ln1 = _ax.plot(x_array, buyer_utility_array, label='buyer_utility')
        ln2 = _ax.plot(x_array, broker_utility_array, label='broker_utility')
        ln3 = _ax.plot(x_array, seller_utility_array, label='seller_utility')
        if nash_point is not None:
            best_variable = nash_point[0]
            nash_buyer_utility, nash_broker_utility, nash_seller_utility = nash_point[1], nash_point[2], nash_point[3]
            _ax.scatter(best_variable, nash_buyer_utility, color='red', label='nash_buyer_utility')
            _ax.scatter(best_variable, nash_broker_utility, color='green', label='nash_broker_utility')
            _ax.scatter(best_variable, nash_seller_utility, color='blue', label='nash_seller_utility')
            _ax.annotate(f'({round(best_variable, 3)}, {round(nash_buyer_utility, 3)})', (best_variable, nash_buyer_utility))
            _ax.annotate(f'({round(best_variable, 3)}, {round(nash_broker_utility, 3)})', (best_variable, nash_broker_utility))
            _ax.annotate(f'({round(best_variable, 3)}, {round(nash_seller_utility, 3)})', (best_variable, nash_seller_utility))

    # 隐藏三个子图之间的坐标轴
    ax.set_frame_on(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax1.set_xticks([])
    ax2.set_xticks([])

    # 设置y轴范围为正值
    ax1.set_ylim(min_buyer_utility, max_buyer_utility)
    ax2.set_ylim(min_broker_utility, max_broker_utility)
    ax3.set_ylim(min_seller_utility, max_seller_utility)

    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)

    ax3.set_xlabel(x_label)
    ax2.set_ylabel('utility')

    ln = ln1 + ln2 + ln3
    labs = [l.get_label() for l in ln]
    plt.legend(ln, labs)
    plt.tight_layout()

def draw_utility_figure_old(x_label, x_array, utility_arrays_list, nash_point, scales, xlims=None, ylims=None, subfigs=False,
                            save_path=None, show_plot=False):
    out_dir = os.path.dirname(save_path) if save_path else 'figures'
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ======= 全局美化设置 =======
    plt.style.use('seaborn-v0_8-colorblind')  # 柔和配色，适合色盲友好
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 15,
        'axes.titlesize': 20,
        'legend.fontsize': 13,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'figure.dpi': 150
    })

    if subfigs:
        xlabel_map = {
            'gamma': r'$\gamma$',
            'q_0': r'$Q_0$',
            'lamda': r'$\lambda$',
            'tao': r'$\tau$',
            't1_t0': r'$\frac{t_1}{t_0}$',
            'data_size': 'Data Size',
        }
        x_label_display = xlabel_map.get(x_label, x_label)
        fig, axs = plt.subplots(3, 1, figsize=(6, 6.5), sharex=True)
        plt.subplots_adjust(hspace=0.6)  # 基准

        labels = ['Consumer Utility', 'TSP Utility', 'Provider Utility']
        colors = ['#E24A33', '#348ABD', '#988ED5']  # red, blue, purpleish

        for i in range(3):
            ax = axs[i]
            for j in range(len(utility_arrays_list)):
                ax.plot(x_array, utility_arrays_list[j][i],
                        label=f'{labels[i]}',
                        color=colors[i],
                        linewidth=2,
                        marker='o' if j == 0 else None,
                        markersize=3)

            if nash_point is not None:
                best_variable = nash_point[0]
                nash_utility = nash_point[i+1]
                ax.scatter(best_variable, nash_utility,
                           color='black', label='Nash Point', zorder=5)
                ax.annotate(f'({best_variable:.4f}, {nash_utility:.2f})',
                            (best_variable, nash_utility),
                            textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10)

            ax.set_ylabel(labels[i])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best', frameon=True, fancybox=True)

            if xlims is not None:
                ax.set_xlim(xlims[i])
            if ylims is not None:
                ax.set_ylim(ylims[i])

        #axs[2].set_xlabel(x_label)
        axs[2].set_xlabel(x_label_display)
        #fig.suptitle(f'Utility under varying {x_label}', fontsize=15)
        axs[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # 添加科学计数法格式化
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图像
        filename = save_path or f'figures/{x_label}_utility_beautified.pdf'
        ext = os.path.splitext(filename)[1].lower()
        fmt = 'png' if ext == '.png' else 'pdf'
        plt.savefig(filename, format=fmt, bbox_inches='tight', dpi=150 if fmt == 'png' else None)
        print(f'[✓] Saved beautified figure to: {filename}')
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        print(f"点的个数: {len(x_array)}")
        return

    raise Exception('Single-plot mode not implemented for beautified version')
'''
def draw_strategy_figure(x_label, x_array, attrs, strategy_arrays_list, scales, xlims=None, ylims=None, subfigs=False):
    (buyer_strategy_array, broker_strategy_arrays, seller_strategy_arrays) = strategy_arrays_list[0]
    fig_num = 1 + len(broker_strategy_arrays) + len(seller_strategy_arrays)
    if subfigs:
        for i in range(fig_num):
            plt.subplot(fig_num, 1, i+1)
            for j in range(len(strategy_arrays_list)):
                (buyer_strategy_array, broker_strategy_arrays, seller_strategy_arrays) = strategy_arrays_list[j]
                if i == 0:
                    strategy_array = buyer_strategy_array
                    label = 'p1'
                elif i < 1 + len(broker_strategy_arrays):
                    strategy_array = broker_strategy_arrays[i-1]
                    label = f'p2_{attrs[i-1]}'
                else:
                    strategy_array = seller_strategy_arrays[i-1-len(broker_strategy_arrays)]
                    label = f'z_{attrs[i-1-len(broker_strategy_arrays)]}'
                plt.plot(x_array, strategy_array)
                plt.xlabel(x_label)
                plt.ylabel(label)
            if xlims is not None:
                plt.xlim(xlims[i])
            if ylims is not None:
                plt.ylim(ylims[i])
        plt.subplots_adjust(hspace=1)
        plt.tight_layout()
        plt.savefig(f'figures/{x_label}_strategy.pdf', format='pdf', bbox_inches='tight')
        #plt.show()
        return
    raise Exception('not implemented')
'''
def draw_strategy_figure(x_label, x_array, attrs, strategy_arrays_list, scales, xlims=None, ylims=None, subfigs=False,
                         save_path=None, show_plot=False):
    # 横坐标标签映射：将 gamma 转换为希腊字母
    xlabel_map = {
        'gamma': r'$\gamma$',
        'q_0': r'$Q_0$',
        'lamda': r'$\lambda$',
        'tao': r'$\tau$',
        't1_t0': r'$t_1/t_0$',
        'data_size': 'Data Size',
    }
    x_label_display = xlabel_map.get(x_label, x_label)

    (buyer_strategy_array, broker_strategy_arrays, seller_strategy_arrays) = strategy_arrays_list[0]
    fig_num = 1 + len(broker_strategy_arrays) + len(seller_strategy_arrays)

    fig_height = 1.5 * fig_num
    fig, axs = plt.subplots(fig_num, 1, figsize=(6, 6.3), sharex=True)
    #plt.subplots_adjust(hspace=0.6 * (3 / fig_num))  # 让 5 张图的空隙跟 3 张图一样紧凑

    # 🎨 多种颜色（可根据需要扩展）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i in range(fig_num):
        ax = axs[i]

        if i == 0:
            strategy_array = buyer_strategy_array
            label = r'$p^S$'
        elif i < 1 + len(broker_strategy_arrays):
            strategy_array = broker_strategy_arrays[i - 1]
            label = f'$p^T_{i}$'
        else:
            strategy_array = seller_strategy_arrays[i - 1 - len(broker_strategy_arrays)]
            label = f'$z_{i - len(broker_strategy_arrays)}$'

        # 🎨 设置颜色，循环使用
        color = colors[i % len(colors)]

        ax.plot(x_array, strategy_array, color=color, linewidth=2)
        ax.set_ylabel(label,fontsize=20)
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', labelsize=15)     # 纵坐标刻度字体同步变大# 纵坐标刻度字体同步变大

        # ✅ 横纵坐标使用科学计数法
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

        ax.grid(True, linestyle='--', alpha=0.5)

        if xlims is not None:
            ax.set_xlim(xlims[i])
        if ylims is not None:
            ax.set_ylim(ylims[i])

    #axs[-1].set_xlabel(x_label)
    axs[-1].set_xlabel(x_label_display, fontsize=20)
    axs[-1].tick_params(axis='x', rotation=45)

    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()

    filename = save_path or f'figures/{x_label}_strategy_sci_colorful.pdf'
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ext = os.path.splitext(filename)[1].lower()
    fmt = 'png' if ext == '.png' else 'pdf'
    plt.savefig(filename, format=fmt, bbox_inches='tight', dpi=150 if fmt == 'png' else None)
    print(f'[✓] Saved colorful strategy figure with scientific notation to: {filename}')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)



def _data_csv_path(filename):
    for base in ('data', '.'):
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Cannot find {filename} under data/ or project root')


def _set_policies_from_p2s(market, p2s):
    for attr_name in market.attributes:
        market.set_p2(attr_name, p2s[attr_name])
    seller_strategy = market.seller_best_strategy()
    for attr_name, strat in seller_strategy.items():
        market.set_policy(attr_name, strat['x'], strat['y'], strat['z'])
    return seller_strategy


def _broker_utility_given_p2s(market, p2s):
    _set_policies_from_p2s(market, p2s)
    return market.broker_utility()


def _estimate_p2_upper(market, attr_name):
    attr = market.attributes[attr_name]
    if not market.cacheEnabled:
        market.build_computation_cache()
    # Provider saturation occurs when z reaches |V_T cap R|; use twice that
    # threshold as a conservative non-negative search bound for Stage 2.
    return max(attr['cache:p2/z'] * max(attr['L_join_B_size'], 1) * 2, 1e-6)


def _broker_best_response(market):
    """TSP best response: analytical only for the linear-quality/log-time benchmark."""
    if (market.decay_type == 'linear' and market.delivery_time_type == 'log'
            and market.risk_cost_type == 'quadratic'):
        return market.broker_best_strategy()
    if not market.cacheEnabled:
        market.build_computation_cache()
    p2s = _optimize_broker_p2s(market)
    seller_strategy = _set_policies_from_p2s(market, p2s)
    return p2s, {name: seller_strategy[name]['z'] for name in market.attributes}


def _optimize_broker_p2s(market, n_p2_points=31, max_rounds=4, refine_rounds=3):
    """Numerically solve Stage 2 for non-linear quality decay.

    We do not reuse the linear closed-form p_T response. Each p_T,i is constrained
    to [0, upper_i], where upper_i is calibrated from the original provider
    saturation threshold. To reduce local-optimum risk, we run multi-start
    coordinate grid search and then shrink the coordinate-wise grid around the
    best point for local refinement. Stage 3 is evaluated with seller_best_strategy(),
    preserving the original discrete rounding and tie-breaking rule.
    """
    if not market.cacheEnabled:
        market.build_computation_cache()
    if market.risk_cost_type != 'quadratic':
        # Non-quadratic risk already recomputes Stage 3 numerically/semianalytically.
        # Use a lighter deterministic coordinate search for calibration speed.
        n_p2_points = min(n_p2_points, 9)
        max_rounds = min(max_rounds, 2)
        refine_rounds = min(refine_rounds, 1)
    attrs = list(market.attributes.keys())
    uppers = {name: _estimate_p2_upper(market, name) for name in attrs}

    def evaluate(p2s):
        return _broker_utility_given_p2s(market, p2s)

    starts = []
    for frac in (0.0, 0.25, 0.5, 1.0):
        starts.append({name: frac * uppers[name] for name in attrs})

    best_global, best_global_u = None, -float('inf')
    for start in starts:
        p2s = dict(start)
        spans = dict(uppers)
        best_u = evaluate(p2s)
        for round_idx in range(max_rounds + refine_rounds):
            improved = False
            for attr_name in attrs:
                center = p2s[attr_name]
                if round_idx < max_rounds:
                    lo, hi = 0.0, uppers[attr_name]
                else:
                    half = max(spans[attr_name] / 2.0, uppers[attr_name] * 1e-6)
                    lo = max(0.0, center - half)
                    hi = min(uppers[attr_name], center + half)
                coord_best, coord_u = center, -float('inf')
                for candidate in np.linspace(lo, hi, n_p2_points):
                    trial = dict(p2s)
                    trial[attr_name] = float(candidate)
                    u = evaluate(trial)
                    if u > coord_u:
                        coord_u, coord_best = u, float(candidate)
                if coord_u > best_u + 1e-9 or abs(coord_best - center) > 1e-9:
                    improved = improved or coord_u > best_u + 1e-9
                    best_u = max(best_u, coord_u)
                    p2s[attr_name] = coord_best
            if round_idx >= max_rounds:
                for attr_name in attrs:
                    spans[attr_name] *= 0.35
            if not improved and round_idx >= max_rounds:
                break
        final_u = evaluate(p2s)
        if final_u > best_global_u:
            best_global_u = final_u
            best_global = dict(p2s)
    return best_global


def _p1_search_hi(market):
    return max(2.0 * market.gamma / max(market.q_0 * 0.01, 1e-6), 5.0)


def _apply_numerical_ret_to_market(market, attrs, ret):
    """Restore the market state to the selected Stackelberg response."""
    market.p1 = float(ret['best_p1'])
    for attr_name, p2 in zip(attrs, ret['best_p2s']):
        market.set_p2(attr_name, float(p2))
    for attr_name, z in zip(attrs, ret['best_zs']):
        attr = market.attributes[attr_name]
        if attr['type'] == 'continuous':
            x = attr['L_size'] - attr['L_join_B_size']
            y = 0
        else:
            x = attr['L_size'] - attr['L_join_B_size']
            y = 0
        market.set_policy(attr_name, x, y, int(z) if attr['type'] == 'discrete' else z)


def _evaluate_p1_candidate(market, attrs, p1):
    market.p1 = float(p1)
    p2s = _optimize_broker_p2s(market)
    seller_strategy = _set_policies_from_p2s(market, p2s)
    broker_u = market.broker_utility()
    buyer_u = market.buyer_utility()
    return {
        'best_p1': float(p1),
        'best_p2s': [p2s[a] for a in attrs],
        'best_zs': [seller_strategy[a]['z'] for a in attrs],
        'buyer_utility': buyer_u,
        'broker_utility': broker_u,
        'seller_utility': market.seller_utility(),
        'quality': market.data_quality(),
    }


def _is_better_stage1_candidate(ret, best, require_broker_positive=True):
    if best is None:
        return True
    if require_broker_positive:
        return ret['buyer_utility'] > best['buyer_utility'] + 1e-9
    return ret['broker_utility'] > best['broker_utility'] + 1e-9


def compute_best_strategy_and_utilities_numerical(market, attrs):
    """Numerical backward induction for non-linear robustness variants.

    Stage 3 reuses the original provider best response because provider utility
    does not directly depend on quality decay or delivery time. When the
    risk-cost function itself changes, Stage 3 is recomputed numerically by
    seller_best_strategy(). Stage 2 uses the bounded multi-start p_T search.
    Stage 1 is a deterministic multi-resolution grid search over the theoretical
    domain p_S >= 0, with broker utility > 0 kept as the feasibility filter.
    """
    market.clean_computation_cache()
    market.build_computation_cache()

    global_hi = _p1_search_hi(market)
    lo, hi = 0.0, global_hi
    if market.risk_cost_type != 'quadratic':
        n_points = 9
        refine_rounds = 3
    else:
        n_points = 25
        refine_rounds = 6
    best_feasible, best_fallback = None, None
    evaluated = {}
    best_round = -1
    last_step = hi / max(n_points - 1, 1)

    def evaluate_once(p1):
        p1 = min(max(float(p1), 0.0), global_hi)
        key = round(p1, 12)
        if key not in evaluated:
            evaluated[key] = _evaluate_p1_candidate(market, attrs, p1)
        return evaluated[key]

    for round_idx in range(refine_rounds + 1):
        candidates = np.linspace(lo, hi, n_points)
        last_step = (hi - lo) / max(n_points - 1, 1)
        round_best = None
        for p1 in candidates:
            ret = evaluate_once(p1)
            if _is_better_stage1_candidate(ret, best_fallback, require_broker_positive=False):
                best_fallback = ret
            if ret['broker_utility'] <= 0:
                continue
            if _is_better_stage1_candidate(ret, best_feasible, require_broker_positive=True):
                best_feasible = ret
                best_round = round_idx
            if _is_better_stage1_candidate(ret, round_best, require_broker_positive=True):
                round_best = ret

        center_ret = best_feasible if best_feasible is not None else best_fallback
        if center_ret is None:
            break
        center = float(center_ret['best_p1'])
        half_width = max(last_step, global_hi * 1e-8)
        lo = max(0.0, center - half_width)
        hi = min(global_hi, center + half_width)
        if hi - lo <= max(global_hi * 1e-10, 1e-10):
            break

    best_ret = best_feasible if best_feasible is not None else best_fallback
    if best_ret is None:
        raise RuntimeError('numerical equilibrium search failed to find a solution')
    if best_feasible is None:
        print('[warn] numerical search: no broker-utility>0 point; using best broker utility fallback')

    near_boundary = float(best_ret['best_p1']) <= max(last_step * 1.5, global_hi * 1e-6)
    best_ret.update({
        'p1_search_method': 'multi_resolution_grid',
        'p1_search_domain': f'[0,{global_hi:.12g}]',
        'p1_search_hi': global_hi,
        'p1_search_rounds': refine_rounds,
        'p1_search_points_per_round': n_points,
        'p1_search_final_step': last_step,
        'p1_boundary_status': 'near-left-boundary' if near_boundary else 'interior',
        'p1_best_round': best_round,
    })
    _apply_numerical_ret_to_market(market, attrs, best_ret)
    return best_ret


def compute_best_strategy_and_utilities(market, attrs):
    """
    计算最优的决策与最优决策下利润。
    对于经纪人与卖家，只计算attrs中的属性的最优决策。
    :param market:
    :param attrs:
    :return: 返回一个字典，包含以下字段：
        1. best_p1: 买家的最优定价p1
        2. best_p2s: 一个数组，为经纪人对attrs中属性的最优定价p2，其中p2的顺序与attrs中属性的顺序一致
        3. best_zs: 一个数组，为卖家对attrs中属性的最优决策z，其中z的顺序与attrs中属性的顺序一致
        4. buyer_utility: 买家的最优利润
        5. broker_utility: 经纪人的最优利润
        6. seller_utility: 卖家的最优利润
    """
    if (market.decay_type != 'linear' or market.delivery_time_type != 'log'
            or market.risk_cost_type != 'quadratic'):
        return compute_best_strategy_and_utilities_numerical(market, attrs)
    return _compute_best_strategy_and_utilities_analytical(market, attrs)


def _average_policy_satisfaction(market):
    psrs = []
    for attr in market.attributes.values():
        if attr['L_size'] > 0:
            psrs.append((attr['x'] + attr['z']) / attr['L_size'])
    return float(np.mean(psrs)) if psrs else 0.0


def solve_stackelberg_numerical(decay_type='exp', eta_scale=2.5, alpha=None,
                                t0_abs=1.0, attrs=None, delivery_time_type='log',
                                delivery_beta=1.5, risk_cost_type='quadratic',
                                delivery_theta1=0.8, delivery_theta2=0.5,
                                delivery_piece_c1_scale=0.5,
                                delivery_piece_c2_scale=1.5,
                                risk_power=2.0, risk_threshold=0.5,
                                risk_threshold_penalty=4.0):
    """Solve the three-stage Stackelberg game and return equilibrium metrics.

    For exp decay, eta_scale is converted to the dimensional eta by
    eta=eta_scale/t0_abs, while q_exp(t)=qmax*exp(-eta*t) uses physical
    delivery_time(). Linear keeps the original analytical benchmark; non-linear
    decay types and non-log delivery-time functions use numerical backward
    induction for Stage 2 and Stage 1.
    """
    if decay_type == 'exp':
        decay_eta = float(eta_scale) / max(float(t0_abs), 1e-12)
        decay_alpha = 2.0 if alpha is None else alpha
    elif decay_type == 'power':
        decay_eta = 0.1
        decay_alpha = 2.0 if alpha is None else alpha
    else:
        decay_eta = 0.1
        decay_alpha = 2.0 if alpha is None else alpha
    market = create_market(decay_type=decay_type, decay_eta=decay_eta,
                           decay_alpha=decay_alpha, t0_abs=t0_abs,
                           delivery_time_type=delivery_time_type,
                           delivery_beta=delivery_beta,
                           delivery_theta1=delivery_theta1,
                           delivery_theta2=delivery_theta2,
                           delivery_piece_c1_scale=delivery_piece_c1_scale,
                           delivery_piece_c2_scale=delivery_piece_c2_scale,
                           risk_cost_type=risk_cost_type,
                           risk_power=risk_power,
                           risk_threshold=risk_threshold,
                           risk_threshold_penalty=risk_threshold_penalty)
    attrs_all = list(market.attributes.keys()) if attrs is None else attrs
    ret = compute_best_strategy_and_utilities(market, attrs_all)
    p2s = ret.get('best_p2s', [])
    zs = ret.get('best_zs', [])
    return {
        'decay_type': decay_type,
        'eta_scale': eta_scale if decay_type == 'exp' else '',
        'eta': decay_eta if decay_type == 'exp' else '',
        'alpha': decay_alpha if decay_type == 'power' else '',
        'delivery_time_type': market.delivery_time_type,
        'delivery_decay_type': market.delivery_time_type,
        'beta': market.delivery_beta if market.delivery_time_type in ('power', 'nonlinear') else '',
        'delivery_theta1': market.delivery_theta1 if market.delivery_time_type in ('piecewise', 'threshold') else '',
        'delivery_theta2': market.delivery_theta2 if market.delivery_time_type in ('piecewise', 'threshold') else '',
        'delivery_piece_c1_scale': market.delivery_piece_c1_scale if market.delivery_time_type in ('piecewise', 'threshold') else '',
        'delivery_piece_c2_scale': market.delivery_piece_c2_scale if market.delivery_time_type in ('piecewise', 'threshold') else '',
        'risk_cost_type': market.risk_cost_type,
        'risk_power': market.risk_power if market.risk_cost_type in ('power', 'cubic') else '',
        'risk_threshold': market.risk_threshold if market.risk_cost_type == 'threshold' else '',
        'risk_threshold_penalty': market.risk_threshold_penalty if market.risk_cost_type == 'threshold' else '',
        't0': market.t0_abs,
        't1': market.t1_abs,
        'pS': ret['best_p1'],
        'pT': ';'.join(f'{v:.8g}' for v in p2s),
        'z': ';'.join(f'{v:.8g}' for v in zs),
        'avg_pT': float(np.mean(p2s)) if p2s else 0.0,
        'avg_psr': _average_policy_satisfaction(market),
        'delivery_time': market.delivery_time(),
        'time_ratio': market.time_ratio(),
        'quality_q': ret['quality'],
        'USC': ret['buyer_utility'],
        'UTSP': ret['broker_utility'],
        'UDP': ret['seller_utility'],
        'p1_search_method': ret.get('p1_search_method', 'analytical'),
        'p1_search_domain': ret.get('p1_search_domain', ''),
        'p1_search_hi': ret.get('p1_search_hi', ''),
        'p1_search_rounds': ret.get('p1_search_rounds', ''),
        'p1_search_final_step': ret.get('p1_search_final_step', ''),
        'p1_boundary_status': ret.get('p1_boundary_status', 'analytical'),
    }


def _compute_best_strategy_and_utilities_analytical(market, attrs):
    ret = {}
    splitters, best_p1_theory, max_buyer_utility_theory, best_discrete_attrs_z = market.buyer_best_strategy()
    # print(f'w: {w}, best_p1_theory: {best_p1_theory}, max_buyer_utility_theory: {max_buyer_utility_theory}')
    best_p1_theory_list = list(best_p1_theory)
    best_p1_theory_list = sorted(best_p1_theory_list)
    best_p1 = best_p1_theory_list[-1]
    ret['best_p1'] = best_p1
    market.p1 = best_p1

    best_p2s, zs = market.broker_best_strategy()
    delta = 0.000000000001 * best_p1
    while True:
        flag = False
        for attr_name in best_discrete_attrs_z:
            if zs[attr_name] != best_discrete_attrs_z[attr_name]:
                flag = True
                break
        if not flag:
            # print(f'actual p1: {market.p1}, delta: {market.p1 - best_p1}')
            break
        market.p1 = best_p1 + delta
        best_p2s, zs = market.broker_best_strategy()
        delta *= 2
        if delta > 0.0000001 * best_p1:
            raise Exception(f'delta too large, bug found!')

    for attr_name in best_p2s:
        market.set_p2(attr_name, best_p2s[attr_name])

    seller_best_strategy = market.seller_best_strategy()
    for attr_name in best_discrete_attrs_z:
        delta = 0.000000000001 * best_p2s[attr_name]
        while seller_best_strategy[attr_name]['z'] != best_discrete_attrs_z[attr_name]:
            market.set_p2(attr_name, best_p2s[attr_name] + delta)
            seller_best_strategy = market.seller_best_strategy()
            delta *= 2
            if delta > 0.0000001 * best_p2s[attr_name]:
                raise Exception(f'delta too large, bug found!')
    for attr_name in seller_best_strategy:
        attr_best_strategy = seller_best_strategy[attr_name]
        market.set_policy(attr_name, attr_best_strategy['x'],
                          attr_best_strategy['y'], attr_best_strategy['z'])
    buyer_utility, broker_utility, seller_utility = market.buyer_utility(), market.broker_utility(), market.seller_utility()
    quality = market.data_quality()
    ret['best_p2s'] = []
    ret['best_zs'] = []
    for attr_name in attrs:
        ret['best_p2s'].append(best_p2s[attr_name])
        ret['best_zs'].append(seller_best_strategy[attr_name]['z'])
    ret['buyer_utility'] = buyer_utility
    ret['broker_utility'] = broker_utility
    ret['seller_utility'] = seller_utility
    ret['quality'] = quality
    return ret


def create_market(decay_type='linear', decay_eta=0.1, decay_alpha=2.0, t0_abs=1.0,
                  delivery_time_type='log', delivery_beta=1.5,
                  delivery_theta1=0.8, delivery_theta2=0.5,
                  delivery_piece_c1_scale=0.5, delivery_piece_c2_scale=1.5,
                  risk_cost_type='quadratic', risk_power=2.0,
                  risk_threshold=0.5, risk_threshold_penalty=4.0):
    seed = random.randint(0, 1000000)
    seed = 1234
    random.seed(seed)
    #data_size=10, lamda=1, q_0= 10, gamma=30, t1_t0=0.5, accuracy=0.8, tao=0
    market = Market(seed=seed, data_size=29649, lamda=0.4, q_0=100, gamma=300, t1_t0=0.5,
                    accuracy=0.8, tao=0.00001, decay_type=decay_type,
                    decay_eta=decay_eta, decay_alpha=decay_alpha, t0_abs=t0_abs,
                    delivery_time_type=delivery_time_type, delivery_beta=delivery_beta,
                    delivery_theta1=delivery_theta1, delivery_theta2=delivery_theta2,
                    delivery_piece_c1_scale=delivery_piece_c1_scale,
                    delivery_piece_c2_scale=delivery_piece_c2_scale,
                    risk_cost_type=risk_cost_type, risk_power=risk_power,
                    risk_threshold=risk_threshold,
                    risk_threshold_penalty=risk_threshold_penalty)
    # market = Market(data_size=5000, lamda=5, q_0=80, gamma=150, t1_t0=0.4, accuracy=0.08, tao=0.001)
    N_attr = 11
    w_min, w_max = 0, 1
    rho_min, rho_max = 0, 1
    #读取min_max_values_Kaggle.csv文件
    df = pd.read_csv(_data_csv_path('min_max_values_kaggle.csv'))
    #把这个文件的第一列存到一个数组中，第二列存到另一个数组中
    min_values = df['min'].values
    max_values = df['max'].values

    df_kaggle = pd.read_csv(_data_csv_path('over-datasets-kaggle-log-clean.csv'))

    i = -1
    for column in df_kaggle.columns:
        i += 1
        # type = 'continuous' if i < N_attr/2+1 else 'discrete'
        w = random.uniform(w_min, w_max)
        rho = random.uniform(rho_min, rho_max)
        if len(df_kaggle[column].value_counts()) < 200:
            type = 'discrete'
            discrete = df_kaggle[column].unique().tolist()
            L_size = random.randint(0,len(df_kaggle[column].value_counts()))
            L = random.sample(discrete,L_size)
            B_size = random.randint(0,len(df_kaggle[column].value_counts()))
            B = random.sample(discrete,B_size)

            set_L = set(L)
            set_B = set(B)
            L_join_B_size = len(set_L & set_B)

        else:
            type = 'continuous'
            L_1 = random.randint(min_values[i], max_values[i])
            L_2 = random.randint(min_values[i], max_values[i])
            L_min = min(L_1, L_2)
            L_max = max(L_1, L_2)
            L_size = L_max - L_min

            B_1 = random.randint(min_values[i], max_values[i])
            B_2 = random.randint(min_values[i], max_values[i])
            B_min = min(B_1, B_2)
            B_max = max(B_1, B_2)
            B_size = B_max - B_min

            L_join_B_start = max(L_min, B_min)
            L_join_B_end = min(L_max, B_max)
            if (L_join_B_start <= L_join_B_end):
                L_join_B_size = L_join_B_end - L_join_B_start
            else:
                L_join_B_size = 0
        market.add_attribute(name=f'A{i}', type=type, w=w, rho=rho, L_size=L_size, B_size=B_size, L_join_B_size=L_join_B_size)
        print(f'A{i}: {type}, w: {w}, rho: {rho}, L_size: {L_size}, B_size: {B_size}, L_join_B_size: {L_join_B_size}')
    """for attr_name in market.attributes:
        print(f'{attr_name}: {market.attributes[attr_name]}')"""
    market.weight_regularize()
    return market


def exp_param(x_label, min, max, N_point, dir):
    market = create_market()
    delta = (max - min) / N_point
    array = [min]
    for i in range(N_point):
        array.append(array[i] + delta)
    attrs = market.attrs_for_test()
    attrs_csv = list(market.attributes.keys())
    buyer_utility_array, broker_utility_array, seller_utility_array = [], [], []
    best_p1_array = []
    best_p2_arrays = [[] for _ in range(len(attrs))]
    best_z_arrays = [[] for _ in range(len(attrs))]
    best_p2_arrays_csv = [[] for _ in range(len(attrs_csv))]
    best_z_arrays_csv = [[] for _ in range(len(attrs_csv))]
    q_array = []
    with tqdm.tqdm(total=len(array)) as pbar:
        for v in array:
            pbar.update(1)
            market.clean_computation_cache()
            if x_label == 'q_0':
                market.q_0 = v
            elif x_label == 'data_size':
                market.data_size = v
            elif x_label == 'gamma':
                market.gamma = v
            elif x_label == 't1_t0':
                market.t1_t0 = v
            elif x_label == 'lamda':
                market.lamda = v
            elif x_label == 'tao':
                market.tao = v
            else:
                raise Exception(f'unknown x_label: {x_label}')
            ret = compute_best_strategy_and_utilities(market, attrs_csv)
            buyer_utility_array.append(ret['buyer_utility'])
            broker_utility_array.append(ret['broker_utility'])
            seller_utility_array.append(ret['seller_utility'])
            q_array.append(ret['quality'])
            best_p1_array.append(ret['best_p1'])
            j = 0
            for i in range(len(attrs_csv)):
                if attrs_csv[i] in attrs:
                    best_p2_arrays[j].append(ret['best_p2s'][i])
                    best_z_arrays[j].append(ret['best_zs'][i])
                    j += 1
                best_p2_arrays_csv[i].append(ret['best_p2s'][i])
                best_z_arrays_csv[i].append(ret['best_zs'][i])

    df = pd.DataFrame()
    df[x_label] = array
    df['buyer_utility'] = buyer_utility_array
    df['broker_utility'] = broker_utility_array
    df['seller_utility'] = seller_utility_array
    df['best_p1'] = best_p1_array
    df['quality'] = q_array
    for i in range(len(attrs_csv)):
        df[f'best_p2_{attrs_csv[i]}'] = best_p2_arrays_csv[i]
        df[f'best_z_{attrs_csv[i]}'] = best_z_arrays_csv[i]
    df.to_csv(f'{dir}/{x_label}.csv', index=False)
    draw_utility_figure_old(x_label=x_label, x_array=array,
                        utility_arrays_list=[[buyer_utility_array, broker_utility_array, seller_utility_array]],
                        nash_point=None, scales=(1, 1, 1), subfigs=True)
    draw_strategy_figure(x_label=x_label, x_array=array, attrs=attrs,
                         strategy_arrays_list=[[best_p1_array, best_p2_arrays, best_z_arrays]],
                         scales=(1, 1, 1), subfigs=True)
    print(market.seed)

def exp1():
    market = create_market()
    # 买家策略的变化对买家、卖家、经纪人效用函数的影响
    print("##### begin to draw figure for buyer_best_strategy #####")
    splitters, best_p1_theory, max_buyer_utility_theory, best_discrete_attrs_z = market.buyer_best_strategy()
    for splitter in splitters:
        print(splitter)
    print(f'best_p1_theory: {best_p1_theory}, max_buyer_utility_theory: {max_buyer_utility_theory}')
    best_p1_theory = list(best_p1_theory)
    best_p1_theory = sorted(best_p1_theory)

    N_point = 1000
    buyer_utility_array, broker_utility_array, seller_utility_array = [], [], []
    best_p1, nash_buyer_utility, nash_broker_utility, nash_seller_utility = 0, 0, 0, 0
    p1_min, p1_max = 5, 25
    p1_array = [p1_min]
    for i in range(N_point):
        p1_array.append(p1_array[i] +  (p1_max - p1_min) / N_point)
    p1_array.extend(best_p1_theory)
    p1_array = sorted(p1_array)
    for p1 in p1_array:
        market.p1 = p1
        best_p2s, _ = market.broker_best_strategy()
        for attr in best_p2s:
            market.set_p2(attr, best_p2s[attr])
        best_strategy = market.seller_best_strategy()
        for attr_name in best_strategy:
            attr_best_strategy = best_strategy[attr_name]
            market.set_policy(attr_name, attr_best_strategy['x'],
                              attr_best_strategy['y'], attr_best_strategy['z'])
        buyer_utility, broker_utility, seller_utility = market.buyer_utility(), market.broker_utility(), market.seller_utility()
        if buyer_utility <= 0 or broker_utility <= 0:
            #print(f'p1: {p1}, deal not possible')
            buyer_utility_array.append(0)
            broker_utility_array.append(0)
            seller_utility_array.append(0)
        else:
            buyer_utility_array.append(buyer_utility)
            broker_utility_array.append(broker_utility)
            seller_utility_array.append(seller_utility)
            if buyer_utility_array[-1] > nash_buyer_utility:
                nash_buyer_utility = buyer_utility_array[-1]
                nash_broker_utility = broker_utility_array[-1]
                nash_seller_utility = seller_utility_array[-1]
                best_p1 = p1
    """if not increasing_array(broker_utility_array):
        raise Exception('broker_utility_array not increasing')
    if not increasing_array(seller_utility_array):
        raise Exception('seller_utility_array not increasing')"""
    print(f'best_p1: {best_p1}, nash_buyer_utility: {nash_buyer_utility}, nash_broker_utility: {nash_broker_utility}, nash_seller_utility: {nash_seller_utility}')

    #L-B的大小
    L_B_A0 = market.attributes['A0']['L_size'] - market.attributes['A0']['L_join_B_size']
    L_B_A2 = market.attributes['A2']['L_size'] - market.attributes['A2']['L_join_B_size']

    #p^BQ_0 \frac{t_1}{t_0} \frac{\omega_ic_{1i}}{|V_i^{B\setminus R}|
    market.build_computation_cache()
    abc = market.attributes['A0']['B_size']*market.attributes['A0']['B_size']/(2*market.lamda*market.attributes['A0']['rho']*market.attributes['A0']['L_size'])*market.attributes['A0']['cache:p2/z']
    print(f'abc: {abc}')
    U_buyer_first_order = best_p1*market.q_0*market.t1_t0*market.attributes['A0']['w']/market.attributes['A0']['cache:p2/z']/L_B_A0 - market.data_size*L_B_A0/market.attributes['A0']['L_size']
    print(f'U_buyer_first_order: {U_buyer_first_order}')

    abc = market.attributes['A2']['B_size'] * market.attributes['A2']['B_size'] / (2 * market.lamda * market.attributes['A2']['rho'] * market.attributes['A2']['L_size'])*market.attributes['A2']['cache:p2/z']
    print(f'abc: {abc}')
    U_buyer_first_order = best_p1 * market.q_0 * market.t1_t0 * market.attributes['A2']['w'] / market.attributes['A2'][
        'cache:p2/z'] / L_B_A2 - market.data_size * L_B_A2 / market.attributes['A2']['L_size']
    print(f'U_buyer_first_order: {U_buyer_first_order}')

    print(f'best_p1:{best_p1}')
    print(f'lamda:{market.lamda}')
    print(f'q_0:{market.q_0}')
    print(f'w:{market.attributes["A0"]["w"]}')
    print(f'rho:{market.attributes["A0"]["rho"]}')
    print(f'L_B:{L_B_A0}')
    print(f'B_size:{market.attributes["A0"]["B_size"]}')
    print(f'L_size:{market.attributes["A0"]["L_size"]}')
    hope_p1 = market.data_size*L_B_A0/market.attributes['A0']['L_size']/(market.q_0*market.t1_t0*market.attributes['A0']['w']/market.attributes['A0']['cache:p2/z']/L_B_A0)
    print(f'A0_hope_p1:{hope_p1}')

    A2_hope_p1 = market.data_size * L_B_A2 / market.attributes['A2']['L_size'] / (
                market.q_0 * market.t1_t0 * market.attributes['A2']['w'] / market.attributes['A2']['cache:p2/z'] / L_B_A2)
    A2_hope_p1_accurate=0.5*(L_B_A2+1)/market.attributes['A2']['cache:a/p1']/math.log(1+1/L_B_A2,math.e)
    print(f'A2_hope_p1:{A2_hope_p1}')
    print(f'A2_hope_p1_accurate:{A2_hope_p1_accurate}')

    data = {
        'p1': p1_array,
        'buyer_utility': buyer_utility_array,
        'broker_utility': broker_utility_array,
        'seller_utility': seller_utility_array
    }
    df = pd.DataFrame(data)

    csv_path = 'exp_results/buyer_best_strategy.csv'
    df.to_csv(csv_path, index=False)
    print(f'write to {csv_path}')


    draw_utility_figure_old(x_label=r'$p^S$', x_array=p1_array,
                        utility_arrays_list=[[buyer_utility_array, broker_utility_array, seller_utility_array]],
                        nash_point=(best_p1, nash_buyer_utility, nash_broker_utility, nash_seller_utility),
                        scales=(1, 8, 25), subfigs=True)

    # 经纪人策略的变化对买家、卖家、经纪人效用函数的影响
    print("\n##### begin to draw figure for broker_best_strategy #####")
    best_p1 = best_p1_theory[-1]
    market.p1 = best_p1
    best_p2s, zs = market.broker_best_strategy()
    delta = 0.000000000001 * best_p1
    while True:
        flag = False
        for attr_name in best_discrete_attrs_z:
            if zs[attr_name] != best_discrete_attrs_z[attr_name]:
                flag = True
                break
        if not flag:
            print(f'actual p1: {market.p1}, delta: {market.p1 - best_p1}')
            break
        market.p1 = best_p1 + delta
        best_p2s, zs = market.broker_best_strategy()
        delta *= 2
        if delta > 0.0000001 * best_p1:
            raise Exception('delta too large, bug found!')
    print(f'best_p2s_theory: {best_p2s}')
    attrs = market.attrs_for_test()
    N_point = 10000
    for i in range(len(attrs)):
        p2_min, p2_max = [0, 0.005] if i == 0 else [0, 0.002]
        p2_array = [p2_min]
        for j in range(N_point):
            p2_array.append(p2_array[j] + (p2_max - p2_min) / N_point)
        p2_array.append(best_p2s[attrs[i]])
        p2_array = sorted(p2_array)

        buyer_utility_array, broker_utility_array, seller_utility_array = [], [], []
        best_p2, nash_buyer_utility, nash_broker_utility, nash_seller_utility = 0, 0, 0, 0
        for attr_name in best_p2s:
            market.set_p2(attr_name, best_p2s[attr_name])
        cur_attr_name = attrs[i]
        if i == 0:
            xlim=(0, 0.01)
            ylim=(0, 50)
        else:
            xlim = (0, 0.01)
            ylim = (23, 24)
        for p2 in p2_array:
            market.set_p2(cur_attr_name, p2)
            seller_best_strategy = market.seller_best_strategy()
            for attr_name in seller_best_strategy:
                attr_best_strategy = seller_best_strategy[attr_name]
                market.set_policy(attr_name, attr_best_strategy['x'], attr_best_strategy['y'], attr_best_strategy['z'])
            buyer_utility, broker_utility, seller_utility = market.buyer_utility(), market.broker_utility(), market.seller_utility()
            if buyer_utility <= 0 or broker_utility <= 0 or seller_utility <= 0:
                #print(f'p1: {p1}, deal not possible')
                buyer_utility_array.append(buyer_utility)
                broker_utility_array.append(broker_utility)
                seller_utility_array.append(seller_utility)
            else:
                buyer_utility_array.append(buyer_utility)
                broker_utility_array.append(broker_utility)
                seller_utility_array.append(seller_utility)
                if broker_utility_array[-1] > nash_broker_utility:
                    nash_buyer_utility = buyer_utility_array[-1]
                    nash_broker_utility = broker_utility_array[-1]
                    nash_seller_utility = seller_utility_array[-1]
                    best_p2 = p2
        print(f'best_p2_{cur_attr_name}: {best_p2}, nash_buyer_utility: {nash_buyer_utility}, nash_broker_utility: {nash_broker_utility}, nash_seller_utility: {nash_seller_utility}')
        draw_utility_figure_old(x_label=f'$p^T_{i+1}$', x_array=p2_array,
                            utility_arrays_list=[[buyer_utility_array, broker_utility_array, seller_utility_array]],
                            nash_point=(best_p2, nash_buyer_utility, nash_broker_utility, nash_seller_utility),
                            scales=(0.7, 8, 20), subfigs=True)

    # 卖家策略的变化对买家、卖家、经纪人效用函数的影响
    print("\n##### begin to draw figure for seller_best_strategy #####")
    best_p2s, zs = market.broker_best_strategy()
    print(f'zs_theory: {zs}')
    for attr_name in best_p2s:
        market.set_p2(attr_name, best_p2s[attr_name] + 0.00000000001 * best_p2s[attr_name])
    seller_best_strategy = market.seller_best_strategy()
    print(f'seller_best_strategy_theory: {seller_best_strategy}')
    for attr_name in market.attributes:
        attr = market.attributes[attr_name]
        if attr['type'] == 'discrete':
            delta = 0.000000000001 * best_p2s[attr_name]
            while seller_best_strategy[attr_name]['z'] != zs[attr_name]:
                market.set_p2(attr_name, best_p2s[attr_name] + delta)
                seller_best_strategy = market.seller_best_strategy()
                delta = delta * 2
                if delta > 0.0000001 * best_p2s[attr_name]:
                    raise Exception(f'can not find the best z for {attr_name}')
            print(f'actual p2 for {attr_name}: {best_p2s[attr_name] + delta}, delta: {attr["p2"] - best_p2s[attr_name]}')

    N_point = 10000
    for i in range(len(attrs)):
        z_min, z_max = 0, market.attributes[attrs[i]]['L_join_B_size']
        # z_min, z_max = [0, 1] if i == 0 else [0, 10]
        z_array = [z_min]
        attr = market.attributes[attrs[i]]
        if attr['type'] == 'continuous':
            for j in range(N_point):
                z_array.append(z_array[j] + (z_max - z_min) / N_point)
        else:
            for j in range(z_max):
                z_array.append(j + 1)

        buyer_utility_array, broker_utility_array, seller_utility_array = [], [], []
        best_z, nash_buyer_utility, nash_broker_utility, nash_seller_utility = 0, 0, 0, 0
        for attr_name in seller_best_strategy:
            attr_best_strategy = seller_best_strategy[attr_name]
            market.set_policy(attr_name, attr_best_strategy['x'], attr_best_strategy['y'], attr_best_strategy['z'])

        if i == 0:
            xlim = (0, z_max)
            ylim = (7, 8)
        else:
            xlim = (0, z_max)
            ylim = (7.68, 7.75)
        attr_name = attrs[i]
        for z in z_array:
            market.set_policy(attr_name, seller_best_strategy[attr_name]['x'], seller_best_strategy[attr_name]['y'], z)
            buyer_utility, broker_utility, seller_utility = market.buyer_utility(), market.broker_utility(), market.seller_utility()
            if buyer_utility <= 0 or broker_utility <= 0 or seller_utility <= 0:
                # print(f'p1: {p1}, deal not possible')
                buyer_utility_array.append(buyer_utility)
                broker_utility_array.append(broker_utility)
                seller_utility_array.append(seller_utility)
            else:
                buyer_utility_array.append(buyer_utility)
                broker_utility_array.append(broker_utility)
                seller_utility_array.append(seller_utility)
                if seller_utility_array[-1] > nash_seller_utility:
                    nash_buyer_utility = buyer_utility_array[-1]
                    nash_broker_utility = broker_utility_array[-1]
                    nash_seller_utility = seller_utility_array[-1]
                    best_z = z
        print(f'best_z_{attr_name}: {best_z}, nash_buyer_utility: {nash_buyer_utility}, '
              f'nash_broker_utility: {nash_broker_utility}, nash_seller_utility: {nash_seller_utility}')
        draw_utility_figure_old(x_label=f'$z_{i+1}$', x_array=z_array,
                            utility_arrays_list=[[buyer_utility_array, broker_utility_array, seller_utility_array]],
                            nash_point=(best_z, nash_buyer_utility, nash_broker_utility, nash_seller_utility),
                            scales=(1, 7, 30), subfigs=True)
    print(market.seed)
    plt.show()

def exp2():
    market = create_market()
    attrs = market.attrs_for_test()
    w_min, w_max = 0, 1
    rho_min, rho_max = 0, 1
    w_array, rho_array = [], []
    N_point = 100
    for i in range(N_point):
        w_array.append(w_min + (i + 1) * (w_max - w_min) / N_point)
        rho_array.append(rho_min + (i + 1) * (rho_max - rho_min) / N_point)
    for i in range(len(attrs)):
        cur_attr_name = attrs[i]
        for j in range(2):
            if j == 0:
                array = w_array
                label = 'w'
            else:
                array = rho_array
                label = 'rho'
            buyer_utility_array, broker_utility_array, seller_utility_array = [], [], []
            best_p1_array = []
            best_p2_arrays = [[] for _ in range(len(attrs))]
            best_z_arrays = [[] for _ in range(len(attrs))]
            for weight in array:
                market = create_market()
                attr = market.attributes[cur_attr_name]
                attr[label] = weight
                market.weight_regularize(w_only=(j==0), rho_only=(j==1))
                ret = compute_best_strategy_and_utilities(market, attrs)
                buyer_utility_array.append(ret['buyer_utility'])
                broker_utility_array.append(ret['broker_utility'])
                seller_utility_array.append(ret['seller_utility'])
                best_p1_array.append(ret['best_p1'])
                for k in range(len(attrs)):
                    best_p2_arrays[k].append(ret['best_p2s'][k])
                    best_z_arrays[k].append(ret['best_zs'][k])
            draw_utility_figure(x_label=f'{label}_{cur_attr_name}', x_array=array,
                                utility_arrays_list=[[buyer_utility_array, broker_utility_array, seller_utility_array]],
                                nash_point=None, scales=(1, 1, 1), subfigs=True)
            draw_strategy_figure(x_label=f'{label}_{cur_attr_name}', x_array=array, attrs=attrs,
                                 strategy_arrays_list=[best_p1_array, best_p2_arrays, best_z_arrays],
                                 scales=(1, 1, 1), subfigs=True)


def exp3():
    dir = f'exp_results'
    max_id = 1
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        for subdir in os.listdir(dir):
            if os.path.isfile(f'{dir}/{subdir}'):
                continue
            id = int(subdir)
            if id >= max_id:
                max_id = id + 1
    dir = f'{dir}/{max_id}'
    os.makedirs(dir)

    exp_param('data_size', 15000, 30000, 1000, dir)
    exp_param('q_0', 10, 150, 1000, dir)
    exp_param('gamma', 200, 500, 1000, dir)
    #
    exp_param('t1_t0',0.3, 0.5, 1000, dir)
    # #中介的处理成本
    exp_param('tao',0.0001,0.002,1000, dir)
    #卖家的单位风险价格
    exp_param('lamda',0.4,0.8,1000, dir)


def draw():
    x_labels = ['data_size', 'q_0', 'gamma']
    xlims = {
        'data_size': None,
        'q_0': None,
        'gamma': None
    }
    ylims = {
        'data_size': None,
        'q_0': None,
        'gamma': None
    }
    """ylims = {
        'data_size': [(2200, 2600), (100, 220), (20, 45)],
        'q_0': [(1100, 3500), (135, 180), (26, 36)],
        'gamma': [(1300, 10000), (80, 350), (5, 66)]
    }"""
    for x_label in x_labels:
        utility_arrays_list = []
        strategy_arrays_list = []
        array = None
        attrs = None
        for i in range(1, 3):
            df = pd.read_csv(f'exp_results/{i}/{x_label}.csv')
            array = df[x_label]
            attrs = [col.split('_')[-1] for col in df.columns if col.startswith('best_p2_')]
            utility_arrays_list.append([
                df['buyer_utility'],
                df['broker_utility'],
                df['seller_utility']]
            )
            strategy_arrays_list.append([
                df['best_p1'],
                [df[f'best_p2_{attr}'] for attr in attrs],
                [df[f'best_z_{attr}'] for attr in attrs]]
            )
        draw_utility_figure_old(x_label=x_label, x_array=array,
                            utility_arrays_list=utility_arrays_list,
                            nash_point=None, scales=(1, 1, 1), subfigs=True,
                            xlims=xlims[x_label], ylims=ylims[x_label])
        draw_strategy_figure(x_label=x_label, x_array=array, attrs=attrs,
                         strategy_arrays_list=strategy_arrays_list,
                         scales=(1, 1, 1), subfigs=True)




# 运行
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'quality_decay':
        from robustness_quality_decay import run_all_robustness_experiments
        run_all_robustness_experiments(fast=True)
    else:
        #exp1()
        #exp2()
        exp3()


 
