
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from yasiu_math.convolve import moving_average
from yasiu_native.time import measure_real_time_decorator

# from numba import jit, njit
import math
import matplotlib.pyplot as plt


def findPoint(R=10, C=2, alpha_deg=30):
    alfaRadians = alpha_deg * np.pi / 180
    m = np.tan(alfaRadians)
    p1 = np.array((np.cos(alfaRadians) * R, np.sin(alfaRadians) * R + C))

    # Solve quadratic for intersection with Circle 1
    A = 1 + m**2
    B = 2 * m * C
    D = B**2 - 4 * A * (C**2 - R**2)

    if D < 0:
        print("No real intersection.")
        return -1

    sqrtD = np.sqrt(D)
    x1 = (-B + sqrtD) / (2 * A)
    x2 = (-B - sqrtD) / (2 * A)
    y1 = m * x1 + C
    y2 = m * x2 + C

    # Choose intersection in same direction as point on Circle 2
    P2x, P2y = R * math.cos(alfaRadians), C + R * math.sin(alfaRadians)
    chosen = (x1, y1) if x1 * P2x > 0 else (x2, y2)

    return np.sqrt((np.pow(chosen - p1, 2)).sum())


def drawCircles(R, C):
    y_int = C / 2
    x_int = math.sqrt(R**2 - (C**2) / 4)
    points = [(-x_int, y_int), (x_int, y_int)]

    # Generate circle coordinates
    theta = [math.radians(t) for t in range(361)]
    x_circle = [R * math.cos(t) for t in theta]
    y_circle1 = [R * math.sin(t) for t in theta]
    y_circle2 = [C + R * math.sin(t) for t in theta]

    # Plot circles
    plt.figure(figsize=(6, 6))
    plt.plot(x_circle, y_circle1, label="Circle 1 (center (0,0))")
    plt.plot(x_circle, y_circle2, label=f"Circle 2 (center (0,{C}))")


def test1Point():
    RAD = 5
    CHIP = 1
    alfa = 70

    dist = findPoint(R=RAD, C=CHIP, alpha_deg=alfa)
    print(f"Dist: {dist}")

    drawCircles(RAD, CHIP)
    P1 = (np.cos(alfa * np.pi / 180) * RAD, np.sin(alfa * np.pi / 180) * RAD + CHIP)
    plt.title(f"Distance: {dist}")
    plt.scatter(*P1)

    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def CompareCirlceCost():
    RAD = 10
    # CHIP = 2
    for CHIP in [2, 4, 5]:
        plt.figure()
        Alfa = np.linspace(0, 90, 100)
        ResCos = np.cos((90 - Alfa) * np.pi / 180) * CHIP
        ResDist = Alfa * 0

        for ai, al in enumerate(Alfa):
            dt = findPoint(R=RAD, C=CHIP, alpha_deg=al)
            ResDist[ai] = dt

        plt.plot(Alfa, ResDist, label="2CircleDist")
        plt.plot(Alfa, ResCos, label="cos")

        print(ResDist.min() / RAD)
        # plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.title(f"MinDist: {ResDist.min():>3.2f}")
        plt.tight_layout()

    plt.show()


def solve(r, ch, a, b, *, STEP=0.001, originX=None, wRad=None):
    if originX is None:
        initR = np.sin(wRad) * r
        first = np.clip(initR - 2 / r, -r, r)
        end = np.clip(initR + 2 / r, -r, r)
        X = np.arange(first, end, STEP, dtype=np.double)
    else:
        # print(f"Origin: {originX}")
        X = np.linspace(originX - r / 4.5, originX + r / 4.5, 20000, dtype=np.double)
        # X = np.linspace(originX - 1, originX +1, 200)
    # x2 = np.linspace(-90, 90, NUM)
    y1 = a * X + b

    wDeg = X / r * 90
    "Inverse map of X = sin(rad)"
    wDeg = np.arcsin(X / r) / np.pi * 180
    wDeg[np.isnan(wDeg)] = 0
    wRad = np.deg2rad(wDeg)

    # x2 = np.sin(wRad) * r
    # wRad = np.arcsin(wRad)
    Line1y = np.cos(wRad) * r + np.sin(wRad) * (ch / 2) - ch / 2 - ch
    # y2 = np.cos(wRad) * r + np.sin(wRad) * ch - ch / 2
    y2 = Line1y
    # plt.scatter(wDeg / 90 * Radius, y2)
    diff = np.abs(y2 - y1).astype(float)
    diff[np.isnan(diff)] = r
    # plt.plot(wDeg)
    # plt.plot(x, x2, linewidth=3)
    # plt.plot(X, y2, linewidth=3, color='red')
    xind = np.argmin(diff)
    xRes = X[xind]
    yRes = a * xRes + b
    # print(diff)
    # print(f"Min: {diff.min()}")
    # print("Xind:", xind)
    # print(X[[xind, xind]])
    # plt.plot(X, diff, color='green')
    # plt.plot(X[[xind, xind]], [0, 10])
    # plt.plot(X, np.isnan(X), color='black')
    # plt.plot(X)
    # plt.scatter(xRes, yRes, marker='.', s=50)
    return (xRes, yRes)


def findCutDistance(wRad, radius, Chip):
    """"""
    "P1 is previous cut"
    "P2 is current cut"
    P2x = np.sin(wRad) * radius
    P2y = np.cos(wRad) * radius + np.sin(wRad) * (Chip / 2) - Chip / 2
    # plt.scatter(P2x, P2y)

    "P0 is moving center"
    P0x = wRad * 0
    # P0y = np.sin(wRad) * Chip / 2 - Chip / 2
    P0y = (np.sin(wRad) - 1) * Chip / 2
    # plt.scatter(P0x, P0y)
    # plt.scatter(P2x, P2y)
    # plt.plot([P0x, P2x], [P0y, P2y], color='gray', alpha=0.5)

    "Line Intersecting P2 and P0"
    "y = ax + b"
    # wDeg = np.linspace(5, 25, 20)
    a = (P2y - P0y) / (P2x - P0x)
    b = P0y - a * P0x
    x = np.linspace(0, radius, 50)
    LineI = a * x + b
    mask = LineI <= radius
    x = x[mask]
    LineI = LineI[mask]
    # plt.plot(x, LineI, color='green', alpha=0.3)
    # print(a, b)

    # y = np.sin(wRad) * Radius + np.cos(wRad) * Chip
    # P1y = np.sin(wRad) * Radius + np.cos(wRad) * Chip - Chip
    # P1x = (P1y - b) / a
    P1x, P1y = solve(radius, Chip, a, b, wRad=wRad)
    plt.plot([P0x, P1x], [P0y, P1y], color='gray', alpha=0.2)
    P1x, P1y = solve(radius, Chip, a, b, originX=P1x)

    plt.scatter(P1x, P1y, color='lightgreen')
    plt.plot([P1x, P2x], [P1y, P2y], color='orange', alpha=0.8, linewidth=3)

    dist = np.sqrt(np.pow(P1x - P2x, 2) + np.pow(P1y - P2y, 2))
    return dist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cosShifted(x, up, center):
    # offset = np.sin(-x * np.pi / 2) * up
    # return np.cos((x - center) / 2 * np.pi) * (1 - up)
    phase = np.cos(x * np.pi / 2) * center
    # plt.figure()
    # plt.plot(phase)
    # plt.show()
    lift = np.cos(phase) * up
    scaled = np.cos((x - phase) / 2 * np.pi) * (1 - up) + lift
    return scaled


if __name__ == "__main__":
    ""
    wDeg = np.linspace(-70, 80, 200)
    wRad = np.deg2rad(wDeg)
    Flute = 2
    Radius = 5
    ChipRev = 2
    Chip = ChipRev / Flute

    x = np.cos(wRad) * Radius
    y = np.sin(wRad) * Radius
    # plt.plot(x, y, color='black')

    x = np.cos(wRad) * Radius
    y = np.sin(wRad) * Radius
    # plt.plot(x, y + Chip, color='black')

    wDeg = np.linspace(0, 180, 50)
    wRad = np.deg2rad(wDeg)

    "P1 Under"
    x = np.cos(wRad) * Radius
    # y = np.sin(wRad) * Radius + np.sin(np.pi / 2 - wRad) * Chip
    y = np.sin(wRad) * Radius + np.cos(wRad) * (Chip / 2) - Chip / 2
    plt.figure(figsize=(10, 7))
    plt.plot(x, y - Chip, color="green", label="Previous cut")
    plt.plot(x, y, label="Current cut", color='blue', linewidth=2)

    wDeg = 60
    wRad = np.deg2rad(wDeg)
    XDeg = np.linspace(-90, 90, 50)
    # X = np.array([0], dtype=float)

    Y = XDeg * 0
    for wi, w in enumerate(XDeg):
        wRad = np.deg2rad(w)
        dist = findCutDistance(wRad, Radius, Chip)
        Y[wi] = dist

    XRad = np.deg2rad(XDeg)
    tempY = np.cos(XRad) * Radius
    tempX = np.sin(XRad) * Radius
    plt.plot(
        tempX, tempY, color='black', alpha=0.5, dashes=[5, 3],
        label="Stationary circles (for reference)"
    )
    plt.plot(tempX, tempY - Chip, color='black', alpha=0.5, dashes=[5, 3])
    plt.plot(tempX, tempY - 2 * Chip, color='black', alpha=0.5, dashes=[5, 3])
    plt.plot(
        [0, 0], [-Chip, 0], label="ChipLoad per 1 blade ( 180° )",
        linewidth=3, color='red', alpha=0.8
    )
    # plt.plot(XDeg / 90 * Radius, tempY + Chip, label="Cos function", alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    newLine = Line2D([0, 0], [0, 1], color='orange', alpha=1, linewidth=3)
    handles.append(newLine)
    labels.append("Wood thickness")

    plt.grid(True)
    plt.legend(handles=handles, labels=labels, loc='lower right')
    plt.title(f"Cutting comparison, Radius: {Radius}, Chipload: {Chip:<2.2f}")
    plt.xlabel("Y Distance")
    plt.ylabel("X Distance")
    plt.tight_layout()
    # plt.show()

    ""
    # plt.close("all")
    plt.figure()
    # Y = moving_average(Y, 5, "keep")
    plt.plot(XDeg, Y, label="Moving cutter approximation", color='green', linewidth=3)

    XRad = np.deg2rad(XDeg)
    tempY = np.cos(XRad) * Chip  # + np.sin(XRad) * Chip
    plt.plot(XDeg, tempY, color='blue', label="Cos function", alpha=0.5)
    up = 0.05
    center = 0.05
    up = np.tan(Chip / Radius) / 2
    center = -np.tan(Radius / Chip) / 16 / 1.5

    shifted = cosShifted(XDeg / 90, up, center) * Chip
    # plt.plot(XDeg, shifted, label="Shifted", color='red')
    diff = tempY - Y
    plt.grid(True)
    # aprox = np.cos(XRad) + sigmoid(X / 90) / 10

    # plt.figure()
    # plt.plot(X, diff, label="Cos function")

    plt.legend()
    # plt.scatter(8.91, 4.32)
    # plt.scatter(P1x, P1y, color="red")
    # P1xx = np.sin(wRad) * Radius
    # plt.plot([P1xx, P1xx], [0, 5])
    # LineI = P0y + slope * (wDeg - P0x)
    # x = np.cos(wRad) * Radius
    # y = np.sin(wRad) * Radius
    # plt.plot(x, y + np.sin(wRad) * Chip + Chip)
    # plt.close("all")

    plt.grid(True)
    plt.title(f"Model engagement comparison. Chipload: {Chip:<2.2f}")
    plt.xlabel("Degrees")
    plt.tight_layout()
    plt.show()
