import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 

Contour_path = "data/processed/bn_contour.npy"
Output_img = "data/processed/arm_simulation.png"

def main():
    if not Path(Contour_path).exists():
        raise FileNotFoundError (
            f"{Contour_path} not found. Run extract_contour.py first"
        )
    
    pts = np.load(Contour_path) 

    scale = 100.0
    pts_scaled = pts * scale

    x = pts_scaled[:, 0]
    y = pts_scaled[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y, "k--", label="Target Contour")

    arm_x = []
    arm_y = []

    for i in range(len(pts_scaled)):
        arm_x.append(pts_scaled[i, 0])
        arm_y.append(pts_scaled[i, 1])

        ax.clear()
        ax.plot(x, y, "k--", label ="Target Contour")
        ax.plot(arm_x, arm_y, "r-", label ="Arm tip path")

        ax.set_xlim(0, scale)
        ax.set_ylim(0, scale)
        ax.invert_yaxis()
        ax.set_aspect("equal", "box")
        ax.legend()
        plt.pause(0.01)
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Output_img)
    print(f"Saved arm simulation plot to {Output_img}")

if __name__ == "__main__":
    main()
