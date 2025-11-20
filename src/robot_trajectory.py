import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

Contour_path = "data/processed/bn_contour.npy"
Output_img = "data/processed/arm_simulation.png"

SCALE = 100.0 
BASE = np.array([SCALE / 2,SCALE / 2])

L1 = 35.0 
L2 = 35.0
Max_reach = L1 + L2

def ik_2link_plot_coords(x_plot, y_plot, elbow_up=True):
    """
    Inverse kinematics for a 2-link planar arm.
    Input: target in PLOT coordinates (x_plot, y_plot) where y increases downward.
    Output: (theta1, theta2) in radians.
    """

    dx = x_plot - BASE[0]
    dy = y_plot - BASE[1]

   
    dy_math = -dy

    r2 = dx**2 + dy_math**2
    r = np.sqrt(r2)

    if r > Max_reach:
        return None

    cos_theta2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    theta2 = np.arccos(cos_theta2)
    if not elbow_up:
        theta2 = -theta2


    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)

    theta1 = np.arctan2(dy_math, dx) - np.arctan2(k2, k1)

    return theta1, theta2


def fk_2link(theta1, theta2):
    """
    Forward kinematics for the 2-link arm.
    Returns joint positions in PLOT coordinates:
    (base), (joint1), (end_effector)
    """
    
    x0_m, y0_m = 0.0, 0.0

   
    x1_m = L1 * np.cos(theta1)
    y1_m = L1 * np.sin(theta1)

    
    x2_m = x1_m + L2 * np.cos(theta1 + theta2)
    y2_m = y1_m + L2 * np.sin(theta1 + theta2)

    def to_plot(xm, ym):
        
        x = BASE[0] + xm
        y = BASE[1] - ym
        return x, y

    x0, y0 = to_plot(x0_m, y0_m)
    x1, y1 = to_plot(x1_m, y1_m)
    x2, y2 = to_plot(x2_m, y2_m)

    return (x0, y0), (x1, y1), (x2, y2)


def main():
    if not Path(Contour_path).exists():
        raise FileNotFoundError(
            f"{Contour_path} not found. Run extract_contour.py first"
        )

    
    pts = np.load(Contour_path)

    
    pts_scaled = pts * SCALE
    x = pts_scaled[:, 0]
    y = pts_scaled[:, 1]

    fig, ax = plt.subplots()

    ee_path_x = []
    ee_path_y = []

    for i in range(len(pts_scaled)):
        tx, ty = pts_scaled[i]

        
        ik_result = ik_2link_plot_coords(tx, ty, elbow_up=True)
        if ik_result is None:
            
            continue
        theta1, theta2 = ik_result

        
        (x0, y0), (x1, y1), (x2, y2) = fk_2link(theta1, theta2)

        ee_path_x.append(x2)
        ee_path_y.append(y2)

        
        ax.clear()

       
        ax.plot(x, y, "k--", label="Target Contour")

        
        ax.plot([x0, x1, x2], [y0, y1, y2], "-o", label="Robot Arm")

       
        ax.plot(ee_path_x, ee_path_y, "r-", label="End-effector path")

       
        ax.plot(BASE[0], BASE[1], "ko")

        ax.set_xlim(0, SCALE)
        ax.set_ylim(0, SCALE)
        ax.invert_yaxis() 
        ax.set_aspect("equal", "box")
        ax.legend(loc="upper right")

        plt.pause(0.01)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Output_img)
    print(f"Saved arm simulation plot to {Output_img}")


if __name__ == "__main__":
    main()