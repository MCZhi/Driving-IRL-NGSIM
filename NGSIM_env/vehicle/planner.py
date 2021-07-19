import numpy as np
import matplotlib.pyplot as plt
import copy

# QuinticPolynomial
class QuinticPolynomial(object):

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T**4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xe - self.a0 - self.a1*T - self.a2*T**2,
                      vxe - self.a1 - 2*self.a2*T,
                      axe - 2*self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        """
        return the t state based on QuinticPolynomial theory
        """
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    # below are all derivatives
    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt

# QuarticPolynomial
class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6*self.a3 + 24*self.a4 * t

        return xt

# Frenet Path
class FrenetPath:

    def __init__(self):
        # frenet position
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        # global position
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T):
    frenet_paths = []
    DT = 0.1 # time tick [s]

    # Lateral motion planning (Generate path to each offset goal)
    #for di in target_search_area:
    di = target_search_area
    fp = FrenetPath()

    lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, T) # lateral

    fp.t = [t for t in np.arange(0.0, T, DT)]
    fp.d = [lat_qp.calc_point(t) for t in fp.t]
    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

    tv = speed_search_area
    tfp = copy.deepcopy(fp)
    lon_qp = QuarticPolynomial(s_d, s_d_d, s_d_d_d, tv, 0.0, T) # longitudinal

    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

    frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist):
    for fp in fplist:
        for i in range(len(fp.s)):
            fy = fp.d[i]
            fx = fp.s[i]
            fp.x.append(fx)
            fp.y.append(fy)

    return fplist


def planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T):
    fplist = calc_frenet_paths(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_search_area, speed_search_area, T)
    fplist = calc_global_paths(fplist)

    return fplist

if __name__ == "__main__":
    s_d, s_d_d, s_d_d_d = 0, 5, 0 # Longitudinal
    c_d, c_d_d, c_d_dd = 0, 0, 0 # Lateral
    target_search_area, speed_search_area, T  = [0, 12/3.281, -12/3.281], np.linspace(5, 30, 10), 5

    paths = []
    for target in target_search_area:
        for speed in speed_search_area:
            path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target, speed, T)[0]
            paths.append(path)

    plt.figure(figsize=(10,6))
    ax = plt.axes()
    ax.set_facecolor("grey")

    for i in range(len(paths)):
        plt.plot(paths[i].x, paths[i].y)

    plt.plot(np.linspace(0, 90, 50), np.ones(50)*6/3.281, 'w--', linewidth=2)
    plt.plot(np.linspace(0, 90, 50), np.ones(50)*-6/3.281, 'w--', linewidth=2)
    
    plt.show()