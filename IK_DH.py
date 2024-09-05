import numpy as np
from numpy import cos, sin, pi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy as sp
# list of DH parameters
# Structure is d, theta, a, alpha in each row

# The theta angles are also referred to as qi

# Tells the code how many joint values to expect
DOF = 6

q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

# Value of pi from the symbolic library for convenience
spi = sp.pi

# Define DH table

DH_params = []

#print("sin 90 derajat = ", sp.sin(spi/2))
sudut_setup1 = [0.,spi/2., 0., spi/2.,0.,0.]
#sudut_setup2 = [0.,0, 0., 0.,0.,0.]

DH_params.append([178.66, q1, 0, spi/2])
DH_params.append([0, q2, 136.0, 0])
DH_params.append([0, q3, 99.0, 0])
DH_params.append([90.0, q4, 0, spi/2])
DH_params.append([85.15, q5, 0, -spi/2]) #spi/2
DH_params.append([61.90, q6, 0.,  0])
DH_params.append([200, 0, 0, 0])

'''
DH_params.append([19.7452, q1, 0, -spi/2])
DH_params.append([0, q2, 35.796, 0])
DH_params.append([0, q3, 6.4259, -spi/2])
DH_params.append([38.705, q4, 0, spi/2])
DH_params.append([0, q5, 0, -spi/2])
DH_params.append([3.795, q6, 0, 0])
DH_params.append([2.75, 0, 0, 0])
'''

#
# print(DH_params)

# Using the combined DH transformation matrix
def DH_trans_matrix(params):

    d, theta, a, alpha = (params[0], params[1], params[2], params[3])
    #--print("alpha:",alpha)

    mat = sp.Matrix([[sp.cos(theta), -1*sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha),    a*sp.cos(theta)],
                    [sp.sin(theta), sp.cos(theta)*sp.cos(alpha),    -1*sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
                    [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
                    [0,             0,                              0,                              1]])

    return mat

# Get the transformations from the origin to each of the joints and the end effector
def joint_transforms(DH_params):
    transforms = []

    transforms.append(sp.eye(4)) #Assuming the first first joint is at the origin

    for el in DH_params:

        transforms.append(DH_trans_matrix(el))

    return transforms


# To get the jacobain we can use the cross product method since we have all of the transformations

# Get the total transformation to the end effector
# This function gives the symbolic expression for the jacobian
def jacobian_expr(DH_params):

    transforms = joint_transforms(DH_params)
    trans_EF = transforms[0]

    for mat in transforms[1:]:
        trans_EF = trans_EF * mat

    pos_EF = trans_EF[0:3,3]
    J = sp.zeros(6, DOF)

    for joint in range(DOF):
        trans_joint = transforms[0]

        for mat in transforms[1:joint+1]:
            trans_joint = trans_joint*mat

        z_axis = trans_joint[0:3,2]
        pos_joint = trans_joint[0:3,3]

        Jv = z_axis.cross(pos_EF - pos_joint)
        Jw = z_axis

        J[0:3,joint] = Jv
        J[3:6,joint] = Jw

    J = sp.simplify(J)
    return J

# This function evaluates a symbolic jacobian expression using provided joint angles
def jacobian_subs(joints, jacobian_sym):

    # Convert to list if it's an ndarray
    if (isinstance(joints, np.ndarray)):
        joints = joints.flatten().tolist()

    J_l = jacobian_sym

    J_l = J_l.subs(q1, joints[0])
    J_l = J_l.subs(q2, joints[1])
    J_l = J_l.subs(q3, joints[2])
    J_l = J_l.subs(q4, joints[3])
    J_l = J_l.subs(q5, joints[4])
    J_l = J_l.subs(q6, joints[5])

    return J_l

# Verify the two previous functions

# If you're trying to implement the jacobian, you can just take the symbolic jacobian that's printed and hard
# code that into your model

#jacobian_symbolic = jacobian_expr(DH_params)
#print("Jacobia Symbolic: ",jacobian_symbolic)

def trans_EF_eval(joints, DH_params):

    # Convert to list if it's an ndarray
    if (isinstance(joints, np.ndarray)):
        joints = joints.flatten().tolist()

    transforms = joint_transforms(DH_params)
    trans_EF = transforms[0]

    for mat in transforms[1:]:
        trans_EF = trans_EF * mat

    trans_EF_cur = trans_EF

    trans_EF_cur = trans_EF_cur.subs(q1, joints[0])
    trans_EF_cur = trans_EF_cur.subs(q2, joints[1])
    trans_EF_cur = trans_EF_cur.subs(q3, joints[2])
    trans_EF_cur = trans_EF_cur.subs(q4, joints[3])
    trans_EF_cur = trans_EF_cur.subs(q5, joints[4])
    trans_EF_cur = trans_EF_cur.subs(q6, joints[5])

    return trans_EF_cur

# This is just for visualizing the robot

def pose(joints, DH_params):

    # Convert to list if it's an ndarray
    if (isinstance(joints, np.ndarray)):
        joints = joints.flatten().tolist()

    transforms = joint_transforms(DH_params)
    #--print("Transformation: ",transforms)

    trans_EF = trans_EF_eval(joints, DH_params)

    pos_EF = trans_EF[0:3,3]

    xs = []
    ys = []
    zs = []

    J = sp.zeros(6, DOF)

    for joint in range(DOF):

        trans_joint = transforms[0]

        for mat in transforms[1:joint+1]:
            trans_joint = trans_joint*mat

        pos_joint = trans_joint[0:3,3]

        pos_joint = pos_joint.subs(q1, joints[0])
        pos_joint = pos_joint.subs(q2, joints[1])
        pos_joint = pos_joint.subs(q3, joints[2])
        pos_joint = pos_joint.subs(q4, joints[3])
        pos_joint = pos_joint.subs(q5, joints[4])
        pos_joint = pos_joint.subs(q6, joints[5])

        xs.append(pos_joint[0])
        ys.append(pos_joint[1])
        zs.append(pos_joint[2])

    xs.append(pos_EF[0])
    ys.append(pos_EF[1])
    zs.append(pos_EF[2])


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    #print("xs :",xs)
    #print("ys :",ys)
    #print("zs :",zs)


    ax.plot(xs, ys, zs, '-o', markersize=10, markerfacecolor='red')

    ax.set_xlim3d(-500,500)
    ax.set_ylim3d(-500,500)
    ax.set_zlim3d(0,600)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.plot(xs, ys, zs)
    plt.show()

#This is just for testing the robot plotting mechanism

#joints = [0, -pi/2, 0, pi/2, 0, 0]

#plot_pose(joints, DH_params)


# This takes the current joints and saturates them to the joint limits if they're out of bounds
#updated joint limits
def joint_limits(joints):

    # Joint 1
    if (joints[0] < -pi*0.89):
        joints[0] = -pi*0.89

    elif (joints[0] > pi*0.89):
        joints[0] = pi*0.89

    # Joint 2
    if (joints[1] < -pi/2):
        joints[1] = -pi/2

    elif (joints[1] > pi/2):
        joints[1] = pi/2

    # Joint 3
    if (joints[2] < -pi/2):
        joints[2] = -pi/2

    elif (joints[2] > pi/2):
        joints[2] = pi/2

    # Joint 4
    if (joints[3] < -pi/2):
        joints[3] = -pi/2

    elif (joints[3] > pi/2):
        joints[3] = pi/2

    #return joints

    # Joint 5
    if (joints[4] < -pi*0.89):

        joints[4] = -pi*0.89

    elif (joints[4] > 0.89*pi):

        joints[4] = 0.89*pi

    # Joint 6
    if (joints[5] < -0.95*pi):

        joints[5] = -0.95*pi

    elif (joints[5] > 0.95*pi):

        joints[5] = 0.95*pi

    return joints

def i_kine(joints_init, target, DH_params, error_trace=False, no_rotation=False, joint_lims=True):

    joints = joints_init

    xr_desired = target[0:3,0:3]
    xt_desired = target[0:3,3]
    x_dot_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_trace = []
    iters = 0;

    #print("Finding symbolic jacobian")

    # We only do this once since it's computationally heavy
    jacobian_symbolic = jacobian_expr(DH_params)

    #print("Starting IK loop")

    final_xt = 0

    while(iters <= 100):
        jac = jacobian_subs(joints, jacobian_symbolic)
        jac = np.array(jac).astype(np.float64)

        trans_EF_cur = trans_EF_eval(joints, DH_params)
        trans_EF_cur = np.array(trans_EF_cur).astype(np.float64)

        xr_cur = trans_EF_cur[0:3,0:3]
        xt_cur = trans_EF_cur[0:3,3]

        final_xt = xt_cur

        xt_dot = xt_desired - xt_cur

        # Find error rotation matrix
        R = xr_desired @ xr_cur.T

        # convert to desired angular velocity
        v = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1)/2)
        r = (0.5 * sin(v)) * np.array([[R[2,1]-R[1,2]],
                                       [R[0,2]-R[2,0]],
                                       [R[1,0]-R[0,1]]])

        # The large constant just tells us how much to prioritize rotation
        xr_dot = 200 * r * sin(v)

        # use this if you only care about end effector position and not rotation
        if (no_rotation):
            xr_dot = 0 * r

        xt_dot = xt_dot.reshape((3,1))

        x_dot = np.vstack((xt_dot, xr_dot))
        x_dot_norm = np.linalg.norm(x_dot)

        #print(x_dot_norm)

        if (x_dot_norm > 25):
            x_dot /= (x_dot_norm/25)

        x_dot_change = np.linalg.norm(x_dot - x_dot_prev)

        # This loop now exits if the change in the desired movement stops changing
        # This is useful for moving close to unreachable points
        if (x_dot_change < 0.005):
            break;

        x_dot_prev = x_dot

        e_trace.append(x_dot_norm)

        Lambda = 12
        Alpha = 1

        joint_change = Alpha * np.linalg.inv(jac.T@jac + Lambda**2*np.eye(DOF)) @ jac.T @ x_dot

        joints += joint_change

        if (joint_lims): joints = joint_limits(joints)

        iters += 1

    #print("Done in {} iterations".format(iters))

    #print("Final position is:")
    #print(final_xt)

    return (joints, e_trace) if error_trace else joints

def findDegree(x, y, z, currentJ):
    print("X, Y, Z Input : ", x, ";", y, ";", z)
    rotateJ = [0.,0.,0.,0.,0.,0.]
    joints = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])

    target = np.array([[0, 0, 1, x],
                   [0, 1, 0, y],
                   [1, 0, 0, z],
                   [0, 0, 0, 1]])

    new_j, e_trace = i_kine(joints, target, DH_params, error_trace=True)
    #print(type(new_j))
    #print(new_j[0][0], "New_J 0")
    degJ = []
    degJ.append(new_j[0][0])
    degJ.append(new_j[1][0])
    degJ.append(new_j[2][0])
    degJ.append(new_j[3][0])
    degJ.append(new_j[4][0])
    degJ.append(new_j[5][0])
    print(degJ)

    rotateJ[0] = degJ[0] - currentJ[0]
    rotateJ[1] = degJ[1] - currentJ[1]
    rotateJ[2] = degJ[2] - currentJ[2]
    rotateJ[3] = degJ[3] - currentJ[3]
    rotateJ[4] = degJ[4] - currentJ[4]
    rotateJ[5] = degJ[5] - currentJ[5]

    return rotateJ, currentJ, 