# if a trajectory has been given, I want a control{u}.
class env():
  def __init__(self, theta0, thetaf):
    self.theta0 = theta0
    self.thetaf = thetaf
    self.a = 2 # m/s
    self.u = 1  # 一个控制量：法向加速度
    trajectory_
