import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def CalcCutingArcLength(diameter, engage=0.5):
    """"""
    radius = diameter / 2
    "Proper calculation are coded"
    # p = diameter * engage
    # r2 = radius - p
    # angle = np.arccos(r2 / radius)  # <0, 3.14> radians
    # factor = angle / np.pi
    # arcL = 2 * np.pi * radius * factor
    "Reduced version"
    # r2 = diameter / 2 - diameter * engage
    # angle = np.arccos(r2 / radius)
    "Mega reduced"
    r4 = (1 - 2 * engage)
    angle = np.arccos(r4)
    arcL = angle * diameter

    return arcL


# def SingleSliceArea(X, Y1, Y2):
#     YSmol = Y1 if Y1 < Y2 else Y2
#     YDiff = np.abs(Y1 - Y2)
#     return X * YSmol + YDiff * X / 2


def EstimateChipLeftOvers(radius, chipLoad):
    """"""
    # ShapeAREA = EstimateCircleVerticalSlice(diameter / 2, diameter * engage)
    # MissingBox = diameter / 2 * engage * chipLoad
    # return ShapeAREA - MissingBox
    alfa = np.arcsin(chipLoad / radius)
    x = radius * (1 - np.cos(alfa))
    return chipLoad * x / 2


def getCutingAngle(diameter, engage):
    """return Angle in radians"""
    radius = diameter / 2
    p = diameter * engage
    r2 = radius - p
    angle = np.arccos(r2 / radius)  # <0, 3.14> radians
    return angle


def EstimateMoonArcArea(diameter, chipLoad, engage):
    leftOver = EstimateChipLeftOvers(diameter / 2, chipLoad)
    Box = diameter * engage * chipLoad
    radius = diameter / 2
    circle = np.pi * radius * radius * getCutingAngle(diameter, engage) / np.pi

    prevCircle = circle - Box
    return circle - prevCircle + leftOver


def HeatSurface():
    # plt.figure()
    for di, Diameter in enumerate([3, 4, 6]):
        plt.subplot(2, 2, di + 1)
        Engage = np.linspace(0.1, 1, 800, dtype=float)
        Depth = np.linspace(1, 20, 800, dtype=float)
        ChipLoad = np.linspace(0.01, .6, 800, dtype=float)

        # CalcCutingArcLength(Diameter, Engage)
        XEng, YDep = np.meshgrid(Engage, Depth)
        _, YChp = np.meshgrid(Engage, ChipLoad)
        usedCmap = get_cmap('nipy_spectral')
        usedNorm = Normalize(0, 4, False)

        ArcLengths = CalcCutingArcLength(Diameter, XEng)
        AREA = ArcLengths * YDep
        # AREA = ArcLengths * YChp
        plt.contour(YDep, XEng, AREA, levels=10, cmap=usedCmap, linewidths=2)

        # AREA = EstimateMoonArcArea(Diameter, YChp, XEng)
        # plt.contour(YChp, XEng, AREA, levels=10, cmap=usedCmap, linewidths=2, linestyles='dashed')

        plt.ylabel("% Kontaktu frezu")
        plt.xlabel("Głębokość [mm]")
        # plt.xlabel("ChipLoad [mm]")
        plt.grid(True, color="black")
        plt.title(f"Płaszczyzna styku $[mm^2]$, Frez: {Diameter}mm")

        cbr = plt.colorbar()
        cbtks = cbr.get_ticks()
        cbtks = np.linspace(cbtks[0], cbtks[-1], len(cbtks) * 2 - 1)
        cbtksLabels = cbtks.round(2)
        cbr.set_ticks(cbtks)
        cbr.set_ticklabels(cbtksLabels)

        # xtks = np.arange(Engage[0], Engage[-1] + 0.001, 0.1)
        # plt.xticks(xtks)
        # plt.yticks(np.arange(Depth[0], Depth[-1] + 0.001, 0.5))
        # plt.semilogx()

    plt.tight_layout()


def AttackSurface():
    # plt.figure(figsize=(10, 6))
    # for di, Diameter in enumerate([3, 4, 6]):
    # plt.subplot(1, 3, di + 1)
    Engage = np.linspace(0.1, 1, 800, dtype=float)
    Depth = np.linspace(1, 20, 800, dtype=float)
    ChipLoad = np.linspace(0.01, .1, 800, dtype=float)

    # CalcCutingArcLength(Diameter, Engage)
    # XEng, YDep = np.meshgrid(Engage, Depth)
    XDep, YChp = np.meshgrid(Depth, ChipLoad)
    usedCmap = get_cmap('nipy_spectral')
    usedNorm = Normalize(0, 4, False)

    # ArcLengths = CalcCutingArcLength(Diameter, XEng)
    AREA = YChp * XDep
    # AREA = ArcLengths * YChp
    plt.contour(XDep, YChp, AREA, levels=12, cmap=usedCmap, linewidths=2)

    # AREA = EstimateMoonArcArea(Diameter, YChp, XEng)
    # plt.contour(YChp, XEng, AREA, levels=10, cmap=usedCmap, linewidths=2, linestyles='dashed')

    # plt.ylabel("% Kontaktu frezu")
    plt.xlabel("Głębokość [mm]")
    plt.ylabel("ChipLoad [mm]")

    plt.grid(True, color="black")
    plt.title(f"Płaszczyzna Ataku $[mm^2]$")
    for deo in [3.33, 5, 6.66, 10]:
        plt.plot([deo, deo], [ChipLoad[0], ChipLoad[-1]],
                 color='black', alpha=0.7, dashes=[3, 4], linewidth=2)
    cbr = plt.colorbar()
    cbtks = cbr.get_ticks()
    cbtks = np.linspace(cbtks[0], cbtks[-1], len(cbtks) * 2 - 1)
    cbtksLabels = cbtks.round(2)
    cbr.set_ticks(cbtks)
    cbr.set_ticklabels(cbtksLabels)


def ChipSizePlot():
    # plt.figure(figsize=(10, 6))
    # for di, Diameter in enumerate([3, 4, 6]):
    # plt.subplot(1, 3, di + 1)
    Engage = np.linspace(0.01, 0.5, 800, dtype=float)
    # Depth = np.linspace(0.5, 4, 800, dtype=float)
    ChipLoad = np.linspace(0.01, .5, 800, dtype=float)

    # CalcCutingArcLength(Diameter, Engage)
    # XEng, YDep = np.meshgrid(Engage, Depth)
    # XDep, YChp = np.meshgrid(Depth, ChipLoad)
    XEng, YChp = np.meshgrid(Engage, ChipLoad)
    usedCmap = get_cmap('nipy_spectral')
    usedNorm = Normalize(0, 4, False)

    AREA = YChp * np.cos(abs(0.5 - XEng) * np.pi)
    # AREA = ArcLengths * YChp
    # plt.contour(YChp, XEng, AREA, levels=9, cmap=usedCmap, linewidths=2)
    x = np.linspace(0, 0.5, 300)
    y = 100 * np.cos((0.5 - x) * np.pi)
    plt.plot(x, y, linewidth=3, color='red')
    plt.grid(True)

    # AREA = EstimateMoonArcArea(Diameter, YChp, XEng)
    # plt.contour(YChp, XEng, AREA, levels=10, cmap=usedCmap, linewidths=2, linestyles='dashed')

    plt.ylabel("% Chip Loadu przy ataku")
    plt.xlabel("% Kontaktu frezu")

    plt.grid(True, color="black")
    plt.title("ChipLoad wzgledem % kontaktu")

    # xtks = np.arange(Engage[0], Engage[-1] + 0.001, 0.1)
    # plt.xticks(xtks)
    # plt.yticks(np.arange(Depth[0], Depth[-1] + 0.001, 0.5))
    # plt.semilogy()


plt.tight_layout()


if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    HeatSurface()
    plt.subplot(2, 2, 4)
    AttackSurface()
    # plt.figure(figsize=(4,3))
    # ChipSizePlot()

    plt.tight_layout()
    plt.show()
