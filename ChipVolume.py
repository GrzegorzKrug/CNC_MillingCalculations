import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from ChipArea import EstimateMoonArcArea
# from ChipLoad import CalcChipLoad


for Depth in [1]:
    Engage = np.linspace(0.2, 1, 400)
    Diameter = 4
    ChipLoads = np.linspace(0.1, 0.7, 400)

    YY, XX = np.meshgrid(ChipLoads, Engage)
    # AREA = CalcChipAreaSurface(Diameter, XX) * Depth
    AREA = EstimateMoonArcArea(Diameter, YY, XX)

    ChipVolume = AREA * Depth

    usedCmap = get_cmap('nipy_spectral')
    # plt.figure()
    plt.title(f"Wielkość wiórków [$mm^3$], Diameter: {Diameter}, Głębokość:{Depth}mm")
    plt.contour(XX, YY, ChipVolume, levels=9, cmap=usedCmap)
    plt.xlabel("% Kontaktu")
    plt.ylabel("ChipLoad")
    plt.grid(True, color="black")

    cbr = plt.colorbar()

    plt.tight_layout()

plt.show()
