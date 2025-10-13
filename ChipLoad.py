import matplotlib.pyplot as plt
import numpy as np

from matplotlib.style import use
from matplotlib.cm import get_cmap

# use('ggplot')


def CalcChipLoad(feed_mm_min, rpm, flutes=3):
    """
    Chip load = forward distance per cut.
    """
    chip_load_mm = feed_mm_min / (rpm * flutes)

    return chip_load_mm


def reverseChipLoad(feed_mm_min, rpm, flutesFrom, loadFrom, flutesTo):
    pass


if __name__ == "__main__":
    Feed = np.linspace(200, 3000, 100)
    Rpm = np.linspace(8000, 20000, 100)
    F, R = np.meshgrid(Feed, Rpm)

    chipLoad2 = CalcChipLoad(F, R, 2)
    chipLoad3 = CalcChipLoad(F, R, 3)
    chipLoad4 = CalcChipLoad(F, R, 4)
    ColorSteps = 12
    # usedCmap = get_cmap('coolwarm')
    # usedCmap = get_cmap('plasma')
    # usedCmap = get_cmap('turbo')
    # usedCmap = get_cmap('terrain')
    usedCmap = get_cmap('nipy_spectral')

    plt.figure(figsize=(12,8), dpi=120)

    def makeChipLoadPLot(F, R, blades, ColorSteps, style='solid', barAX=None):
        """"""
        plt.grid(True, linestyle='-', color='black', alpha=0.6, linewidth=1)

        chipLoad = CalcChipLoad(F, R, blades)
        MIN, MAX = 0.01, 0.08
        levels = np.linspace(MIN, MAX, ColorSteps + 3)
        # levelsContour = np.linspace(MIN, MAX, ColorSteps * 2 + 1)

        # plt.contourf(F, R chipLoad3, cmap=usedCmap, levels=levels, alpha=0.6)
        ctr_lines = plt.contour(
            F, R, chipLoad, cmap=usedCmap, levels=levels,
            alpha=1, linewidths=4, linestyles=style,
        )

        plt.ylabel("RPM - Obroty / minutę", size=10)
        plt.xlabel("Prędkość [mm/min]", size=10)
        # plt.ylabel("RPM - Obroty / minutę", size=15)
        # plt.xtick
        gca = plt.gca()
        # xtk = gca.get_xticks()
        # ytk = gca.get_yticks()
        plt.xticks(np.arange(Feed[0], Feed[-1] + 1, 200), size=12, rotation=30)
        ytk = np.arange(Rpm[0], Rpm[-1] + 1, 1000)
        plt.yticks(ytk, labels=[f"{round(num / 1000)}k" for num in ytk], size=12)
        # plt.xticks(np.arange(Rpm[0], Rpm[-1] + 1, 1000), size=10)
        # plt.yticks(np.arange(Feed[0], Feed[-1] + 1, 100), size=10)

        plt.title(f"{blades} ostrza", size=13)
        # plt.tight_layout()

        "COLOR BAR"
        if (barAX):
            pass
            cbar = plt.colorbar(ctr_lines, cax=barAX)
            cbar.ax.tick_params(labelsize=13)
            rawTicks = cbar.get_ticks()
            colorTks = np.linspace(rawTicks[0], rawTicks[-1], ColorSteps * 2 + 1)  # Set tick values
            colorTks = rawTicks
            tkLables = colorTks.round(3)  # Set tick labels
            cbar.set_ticks(colorTks)
            cbar.set_ticklabels(tkLables)
        plt.ylim(8000, 16000)
    Nx, Ny = 2, 2
    barAX = plt.subplot(Nx, Ny, 4)

    plt.subplot(Nx, Ny, 1)
    makeChipLoadPLot(F, R, 2, ColorSteps)
    plt.xlim(500, 2000)

    plt.subplot(Nx, Ny, 2)
    makeChipLoadPLot(F, R, 3, ColorSteps)
    # plt.xlim(500, 1500)
    plt.xlim(500, 2000)

    plt.subplot(Nx, Ny, 3)
    makeChipLoadPLot(F, R, 4, ColorSteps, barAX=barAX)
    # plt.xlim(600, 1800)
    plt.xlim(500, 2000)

    plt.subplot(Nx, Ny, 4)
    plt.title("Chip load [mm]", size=14)
    plt.tight_layout()
    plt.show()
