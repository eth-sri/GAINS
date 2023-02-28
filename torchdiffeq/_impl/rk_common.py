import bisect
import collections
import torch
from .event_handling import find_event
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _select_initial_step_mod,
                   _optimal_step_size,
                   _multiplicative_step_size,
                   _UncheckedAssign,
                   _runge_kutta_step,
                   _compute_error_est)
from .misc import Perturb
from .solvers import AdaptiveStepsizeEventODESolver
import pdb


_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.

'''
class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None


def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t0 = t0.to(y0.dtype)
    dt = dt.to(y0.dtype)
    t1 = t1.to(y0.dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = Perturb.PREV
        else:
            ti = t0 + alpha_i * dt
            perturb = Perturb.NONE
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi, perturb=perturb)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + k.matmul(dt * tableau.c_sol).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = k.matmul(dt * tableau.c_error)
    return y1, f1, y1_error, k
'''

# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3, perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeEventODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol,
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 adaptive_step_factor = None,
                 running_mean_step = None,
                 dtype=torch.float64,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device
        self.device = device
        self.list = None
        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype
        self.adaptive_step_factor = adaptive_step_factor
        self.running_mean_step = running_mean_step

        self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype=dtype, device=device)
        self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype=dtype, device=device)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

    def _before_integrate(self, t):
        t0 = t[0]
        f0 = self.func(t[0], self.y0)
        if self.adaptive_step_factor is not None:
            if self.running_mean_step == None:
                first_step = _select_initial_step_mod(self.func, t[0], self.y0, self.order -1, self.atol, self.tableau,self.dfactor,self.ifactor,self.safety,f0)
                if t0 + first_step > t[1]:
                    first_step = t[1] - t0
                    self.running_mean_step = t[1] - t0
                else:
                    self.running_mean_step = first_step.clone()
            else:
                first_step = self.running_mean_step
        elif self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)

        # Handle step_t and jump_t arguments.
        if self.step_t is None:
            step_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            step_t = _sort_tvals(self.step_t, t0)
            step_t = step_t.to(self.dtype)
        if self.jump_t is None:
            jump_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            jump_t = _sort_tvals(self.jump_t, t0)
            jump_t = jump_t.to(self.dtype)
        counts = torch.cat([step_t, jump_t]).unique(return_counts=True)[1]
        if (counts > 1).any():
            raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(bisect.bisect(self.step_t.tolist(), t[0]), len(self.step_t) - 1)
        self.next_jump_index = min(bisect.bisect(self.jump_t.tolist(), t[0]), len(self.jump_t) - 1)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        self.next_t = next_t
        self.n_acc_steps = 0
        self.n_tot_steps = 0
        self.avg_step = 0.0
        self.after_clip_count = 0
        #print(self.rk_state[4])

        while next_t - 1e-9 > self.rk_state.t1:
            assert self.n_tot_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(self.n_tot_steps, self.max_num_steps)
            temp = self.rk_state[4] 
            self.rk_state,accept_step,_,_,_ = self._adaptive_step(self.rk_state)

            if accept_step:
                self.avg_step += temp
                self.n_acc_steps += 1
            self.n_tot_steps+= 1
        try:

            self.avg_step /= self.n_acc_steps
            self.list = [self.n_tot_steps,self.n_acc_steps,self.avg_step.item()]
        except:
            self.list = [1,1,1]
        #total steps; normal accepted; avg step; running mean step;increase accepted; frist increase; change factor; reject after increase

        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _advance_mod(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        self.next_t = next_t
        activations = []
        #pdb.set_trace()
        #self.func.base_func.activations = []
        n_acc_steps = 0
        n_tot_steps = 0
        n_rejected = 0
        avg_step = 0.0
        err_est = 0.0
        self.after_clip_count = 0
        trajectory = []
        n_acc_steps_inc = 0
        first_increase = 0
        change_factor_inc = torch.tensor([]).to(self.device).view(1,-1)
        change_factor_dec = torch.tensor([]).to(self.device).view(1,-1)
        reject_after_increase = 0
        flag_increased = False
        flag_decreased = False
        flag = False
        err_1 = []
        errs = []
        next_stop_true = False
        already_clipped = False
        while next_t - 1e-8 > self.rk_state.t1:
            assert n_tot_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_tot_steps, self.max_num_steps)
            temp = self.rk_state[4]
            if n_tot_steps == 0:
                states = torch.tensor([self.rk_state.t1,self.rk_state.dt]).to(self.device).type(self.dtype).view(1,2)
            else:
                states = torch.cat((states,torch.tensor([self.rk_state.t1,self.rk_state.dt]).to(self.device).type(self.dtype).view(1,2)),dim = 0 )

            self.rk_state,accept_step,err,flag,traj = self._adaptive_step(self.rk_state,flag)
            
            errs.append(err.item())

            if traj >0 and states[-1][-1] > self.rk_state.dt and self.rk_state.dt> 0:
                next_stop_true = True
                already_clipped = True

            if n_tot_steps <= 1:
                err_1.append(err.clone())
            trajectory += [traj]

            if flag_increased:
                change_factor_inc = torch.cat((change_factor_inc, (err/temp_err).view(1,-1)),dim=1)
            if flag_decreased:
                change_factor_dec = torch.cat((change_factor_dec, (temp_err/err).view(1,-1)),dim = 1)

            if accept_step == True:
                #case where we accept and have to clip step_size is considered as accept without doubling
                if (err <= self.adaptive_step_factor ** (-self.order)) and (self.rk_state[4] > temp):
                    n_acc_steps_inc += 1
                    temp_err = err
                    if n_tot_steps == 0:
                        first_increase = 1
                    flag_increased = True
                    flag_decreased = False

                else:
                    n_acc_steps += 1
                    err_est += err
                    flag_increased = False
                    flag_decreased = False

                avg_step += temp
            else:
                temp_err = err
                if flag_increased:
                    reject_after_increase += 1
                flag_increased = False
                flag_decreased = True
                n_rejected += 1
                
            n_tot_steps+= 1
        avg_step = avg_step / (n_acc_steps + n_acc_steps_inc)
        err_est = err_est / n_acc_steps

        n_acc_steps = trajectory.count(1)
        n_acc_steps_inc = trajectory.count(2)
        n_rejected = trajectory.count(0)

        #total steps; normal accepted; avg step;increase accepted; frist increase; change factor; reject after increase, normal error est; running_mean
        if len(activations) > 0:
            self.list = [n_tot_steps,n_acc_steps,n_acc_steps_inc,n_rejected,avg_step.item(),first_increase,change_factor_inc.mean().item(),change_factor_dec.mean().item(),reject_after_increase,err_est.item(),activations,trajectory,states,self.running_mean_step,err_1]
        else:
            self.list = [n_tot_steps,n_acc_steps,n_acc_steps_inc,n_rejected,avg_step.item(),first_increase,change_factor_inc.mean().item(),change_factor_dec.mean().item(),reject_after_increase,err_est.item(),trajectory,states,self.running_mean_step,err_1]

        return self.rk_state[0]

    def _advance_traj(self, traversed_states):
        """Interpolate through the next time point, integrating as necessary."""
        y1 = self.y0
        f1 = self.func(traversed_states[0,0], self.y0)
        for i in range(traversed_states.shape[0]):
            #handle rejected step without any computations
            if i <= (traversed_states.shape[0] -2):
                if traversed_states[i,0] == traversed_states[i+1,0]:
                    continue
            y0, f0 = y1,f1
            t0, dt = traversed_states[i,0],traversed_states[i,1]
            t1 = t0 + dt

            y1, f1, _, _ = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)

        return y1

    def _advance_until_event(self, event_fn):
        """Returns t, state(t) such that event_fn(t, state(t)) == 0."""
        if event_fn(self.rk_state.t1, self.rk_state.y1) == 0:
            return (self.rk_state.t1, self.rk_state.y1)

        n_steps = 0
        sign0 = torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1))
        while sign0 == torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1)):
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        interp_fn = lambda t: _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, t)
        return find_event(interp_fn, sign0, self.rk_state.t0, self.rk_state.t1, event_fn, self.atol)

    def _adaptive_step(self, rk_state,flag = False):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        t1 = t0 + dt
        # dtypes: self.y0.dtype (probably float32); self.dtype (probably float64)
        # used for state and timelike objects respectively.
        # Then:
        # y0.dtype == self.y0.dtype
        # f0.dtype == self.y0.dtype
        # t0.dtype == self.dtype
        # dt.dtype == self.dtype
        # for coeff in interp_coeff: coeff.dtype == self.y0.dtype

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################

        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            on_step_t = t0 < next_step_t < t0 + dt
            if on_step_t:
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            on_jump_t = t0 < next_jump_t < t0 + dt
            if on_jump_t:
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # Must be arranged as doing all the step_t handling, then all the jump_t handling, in case we
        # trigger both. (i.e. interleaving them would be wrong.)

        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)

        # dtypes:
        # y1.dtype == self.y0.dtype
        # f1.dtype == self.y0.dtype
        # y1_error.dtype == self.dtype
        # k.dtype == self.y0.dtype

        ########################################################
        #                     Error Ratio                      #
        ########################################################

        #reshape y1_error to B, x1,x2,x3 in latent ODE architecture
        if flag:
            accept_step = True
            error_ratio = torch.zeros_like(dt).squeeze()
        else:
            if self.rtol != 0:
                #standard AS 
                error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
            else:
                #CAS
                error_ratio = _compute_error_est(y1_error, self.atol,True).mean()
            accept_step = error_ratio <= 1

        ########################################################
        #                   Update RK State                    #
        ########################################################

        if accept_step:
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                f1 = self.func(t_next, y_next, perturb=Perturb.NEXT)
            f_next = f1
        else:
            t_next = t0
            y_next = y0
            f_next = f0

        if flag:
            #either clipp or remain the same
            dt_next = torch.minimum(dt, self.next_t - t_next)
            trajectory = 1
        else:
            if self.adaptive_step_factor is None:
                dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
                flag = False
                trajectory = 1
            else:
                dt_next,flag,trajectory,self.after_clip_count = _multiplicative_step_size(dt,error_ratio,t_next,self.next_t,self.order,self.after_clip_count,self.adaptive_step_factor)
        
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state,accept_step,error_ratio,flag,trajectory

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0)
        y_mid = y0 + k.matmul(dt * self.mid).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _sort_tvals(tvals, t0):
    tvals = tvals[tvals >= t0]
    return torch.sort(tvals).values
