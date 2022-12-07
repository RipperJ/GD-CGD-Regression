import sys
import logging
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import glob
from PIL import Image

def loggingSetup():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s: %(funcName)25s() ] %(message)s")
    
    info_file_handler = logging.FileHandler(filename='GD-CGD.log', mode='w')
    info_file_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    handlers = [info_file_handler, stdout_handler]
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler) 
    return root

def load_data(file):
    # Reading data from txt file
    data_f = open(file, "r")
    data_f.readline()
    data = data_f.readlines()
    data = [[float(__) for __ in _.split()] for _ in data]
    x1 = np.array([[_[0]] for _ in data])
    x2 = np.array([[_[1]] for _ in data])
    y  = np.array([_[2] for _ in data])
    
    # x1 = (x1 - x1.mean())/x1.std()
    # x2 = (x2 - x2.mean())/x2.std()
    # plt.scatter(x1, y, s=5, label="x1")
    # plt.scatter(x2, y, s=5, label="x2")
    # plt.legend(fontsize=15)
    # plt.xlabel('xs', fontsize=15)
    # plt.ylabel('y', fontsize=15)
    # plt.title("Relation between y and x1, x2")
    # plt.legend()

    # plt.savefig("test.png")
    
    # Adding column of ones to the X vector
    x = np.c_[np.ones(x1.shape[0]), x1, x2] # shape: (1000, 3)
    return x, y

def gradient_descent(x, y, N, theta, alpha):
    cost_list = []          # record all the cost values
    theta_list = []         # record all the theta_0 and theta_1 values 
    vis_theta_list = []     # record all the theta_0 and theta_1 values (only when % 1e4 == 0)
    prediction_list = []
    run = True
    cost_list.append(1e10)  # append large value to the cost list
    i = 0
    while run:
        prediction = np.dot(x, theta)   # predicted y values theta_0*x1 + theta_1*x2
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*N) * np.dot(error.T, error)   #  (1/2m) * sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (1/N) * np.dot(x.T, error))   # alpha * (1/N) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   # convergence check
            run = False
        if i % 10000 == 0:
            logging.info("Iter: {:<6}, cost: {:.6f}".format(i, cost))
            vis_theta_list.append((i, theta, cost))
        i += 1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    vis_theta_list.append((i, theta_list[-1], cost_list[-1]))
    return prediction_list, cost_list, theta_list, vis_theta_list

def conjugate_gradient_descent(x, y, N, theta, alpha):
    theta_list = []
    cost_list = []
    prediction_list = []
    vis_theta_list = []
    run = True
    k = 0
    A = np.dot(x.T, x)
    b = np.dot(x.T, y)
    r = b - np.dot(A, theta)
    p = r
    theta_list.append(theta)
    cost_list.append(r)
    prediction_list.append(np.dot(x, theta))
    vis_theta_list.append((k, theta_list[-1], LA.norm(cost_list[-1])))
    while run:
        alpha_ = LA.norm(cost_list[-1])**2 / np.dot(np.dot(p.T, A), p)
        theta_list.append(theta_list[-1] + alpha_ * p)
        vis_theta_list.append((k, theta_list[-1], LA.norm(cost_list[-1])))
        prediction_list.append(np.dot(x, theta_list[-1]))
        if LA.norm(alpha_ * p) < 1e-9:
            break
        r_new = cost_list[-1] - alpha_ * np.dot(A, p)
        beta_ = LA.norm(r_new)**2 / LA.norm(cost_list[-1])**2
        p = r_new + beta_ * p
        cost_list.append(r_new)
        if r_new.any() == False or p.any() == False or k == N - 2:
            break
        else:
            k += 1
            continue
    alpha_ = LA.norm(cost_list[-1])**2 / np.dot(np.dot(p.T, A), p)
    theta_list.append(theta_list[-1] + alpha_ * p)
    vis_theta_list.append((k, theta_list[-1], LA.norm(cost_list[-1])))
    prediction_list.append(np.dot(x, theta_list[-1]))
    return prediction_list, cost_list, theta_list, vis_theta_list
        

if __name__ == "__main__":
    loggingSetup()
    
    # Load data from txt
    x, y = load_data("data.txt")
    
    # Parameters for GD
    alpha = 0.0001  # learning rate
    N = x.shape[0]  # sample number
    np.random.seed(0)
    theta = np.random.rand(3)   # random guess of initial values
    
    
    
    # 1. Gradient Descent
    logging.info("\n1. Running Gradient Descent (GD) =====================")
    prediction_list, cost_list, theta_list, vis_theta_list = gradient_descent(x, y, N, theta, alpha)
    GD_theta = theta_list[-1]
    yp = GD_theta[0] + GD_theta[1] * x[:,1] + GD_theta[2] * x[:,2]

    # MSE Computation
    plt.title("GD Cost Function J")
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost")
    plt.plot(cost_list)
    logging.info("Generating convergence trajectory (cost) in GD-cost-convergence.png...")
    plt.savefig("GD-cost-convergence.png")
    plt.close()
    MSE_GD = ((prediction_list[-1]-y)**2).mean()
    logging.info('Mean Square Error: {:.6f}'.format(MSE_GD))
    
    # Convergence Animation
    logging.info("Generating GD result animation (GD-result-3d.gif)...")
    for i, t, c in vis_theta_list:
        x1_, x2_ = np.meshgrid((0, 1), (0, 1))
        y_ = t[0] + t[1] * x1_ + t[2] * x2_
        
        ax = plt.figure(dpi=300).add_subplot(projection="3d")
        ax.scatter(x[:,1], x[:,2], yp, c="lightblue")
        ax.plot_surface(x1_, x2_, y_, alpha=0.5, color="orange")
        ax.view_init(elev=10, azim=15)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        plt.title("Iter={}, Cost={:.6f}\ny={:.4f}+{:.4f}x1+{:.4f}x2".format(i, c, t[0], t[1], t[2]))
        plt.savefig("result/GD{:0>6}.png".format(i))
        plt.close()
        
    frames = []
    imgs = sorted(glob.glob("./result/GD*.png"))
    for img in imgs:
        frames.append(Image.open(img))
    frames[0].save("GD-result-3d.gif", format="GIF", append_images=frames[1:], save_all=True, duration=150, loop=0)
    
    # 2. Conjugate Gradient Descent
    logging.info("\n2. Running Conjugate Gradient Descent (CGD) =====================")
    prediction_list, cost_list, theta_list, vis_theta_list = conjugate_gradient_descent(x, y, N, theta, alpha)
    CGD_theta = theta_list[-1]
    yp = CGD_theta[0] + CGD_theta[1] * x[:,1] + CGD_theta[2] * x[:,2]

    # MSE Computation
    plt.title("CGD Cost Function J")
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost")
    plt.plot([LA.norm(_) for _ in cost_list])
    logging.info("Generating convergence trajectory (cost) in CGD-cost-convergence.png...")
    plt.savefig("CGD-cost-convergence.png")
    plt.close()
    MSE_GD = ((prediction_list[-1]-y)**2).mean()
    logging.info('Mean Square Error: {:.6f}'.format(MSE_GD))
    
    # Convergence Animation
    logging.info("Generating CGD result animation (CGD-result-3d.gif)...")
    for i, t, c in vis_theta_list:
        x1_, x2_ = np.meshgrid((0, 1), (0, 1))
        y_ = t[0] + t[1] * x1_ + t[2] * x2_
        
        ax = plt.figure(dpi=300).add_subplot(projection="3d")
        ax.scatter(x[:,1], x[:,2], yp, c="lightblue")
        ax.plot_surface(x1_, x2_, y_, alpha=0.5, color="orange")
        ax.view_init(elev=10, azim=15)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        plt.title("Iter={}, Cost={:.6f}\ny={:.4f}+{:.4f}x1+{:.4f}x2".format(i, c, t[0], t[1], t[2]))
        plt.savefig("result/CGD{:0>6}.png".format(i))
        plt.close()
        
    frames = []
    imgs = sorted(glob.glob("./result/CGD*.png"))
    for img in imgs:
        frames.append(Image.open(img))
    frames[0].save("CGD-result-3d.gif", format="GIF", append_images=frames[1:], save_all=True, duration=500, loop=0)
    
    # Compare Results
    logging.info("\n3. Comparing Results =====================")
    real_theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    logging.info("\n3.1 By Calculating The Inverse Matrix:\ny = {:.4f} + {:.4f}*x1 + {:.4f}*x2".format(real_theta[0], real_theta[1], real_theta[2]))
    logging.info("\n3.2 By Gradient Descent:\ny = {:.4f} + {:.4f}*x1 + {:.4f}*x2".format(GD_theta[0], GD_theta[1], GD_theta[2]))
    logging.info("\n3.3 By Conjugate Gradient Descent:\ny = {:.4f} + {:.4f}*x1 + {:.4f}*x2".format(CGD_theta[0], CGD_theta[1], CGD_theta[2]))