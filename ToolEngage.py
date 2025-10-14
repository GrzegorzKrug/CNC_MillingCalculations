import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from yasiu_math.convolve import moving_average
from yasiu_native.time import measure_real_time_decorator

from numba import jit, njit


@njit(cache=True)
def engageAsSpining(x, up, center):
    # offset = np.sin(-x * np.pi / 2) * up
    # return np.cos((x - center) / 2 * np.pi) * (1 - up)
    phase = np.cos(x * np.pi / 2) * center
    # plt.figure()
    # plt.plot(phase)
    # plt.show()
    lift = np.cos(phase) * up
    scaled = np.cos((x - phase) / 2 * np.pi) * (1 - up) + lift
    return scaled


@njit(cache=True)
def CalcEngageAreaAngle(degreesIn, omega, engage, useClimb=True):
    thAngle = np.sin((0.5 - engage) * np.pi) * 90
    # print(engage, th)
    degrees = degreesIn - omega
    degrees = degrees % 360
    # degHalf = degrees % 180

    if useClimb:
        # if (chip and radius):
        #     up = np.tan(chip / radius) / 2
        #     center = -np.tan(radius / chip) / 16 / 1.5
        #     area = engageAsSpining(degrees / 90, up, center)
        # else:
        area = np.cos(degrees * np.pi / 180)

        if (engage <= 0.5):
            "UNDER 50"
            area[degrees < (thAngle)] = 0
            area[degrees > 90] = 0
        else:
            "OVER 50"
            halfmaskPostcut = (degrees % 180) > 90
            mask2 = degrees < 360 - abs(thAngle)
            area[halfmaskPostcut & mask2] = 0

    else:
        area = np.sin(degrees * np.pi / 180)
        # degreesCalc = (degreesIn - omega -90)  # TO USE SIMILAR MASKS, shifting to cosinus plot
        if (engage <= 0.5):
            ""
            area[(degrees % 180) > abs(90 - thAngle)] = 0
        else:
            area[(degrees % 180) > (90 + abs(thAngle))] = 0
            # area[degHalf > (90 + abs(thAngle))] = 0
        # area[degHalf > 90] = 0

    area[area < 0] = 0
    return area


@njit()
def getCombined(rotation, FLUTES, DEPTH, CYCLE_DEPTH, ENGAGE=0.5, useClimb=True, resolution=0.25):
    angleStep = 360 / FLUTES
    endAngle = DEPTH / CYCLE_DEPTH * 360
    combinedForces = rotation * 0

    for flut in range(FLUTES):
        OMEGA = np.arange(0, endAngle + 0.01, resolution)
        for omega in OMEGA:
            curFor = CalcEngageAreaAngle(
                rotation, omega + flut * angleStep, ENGAGE,
                useClimb=useClimb,
                #   chip=chip, radius=radius
            ) / len(OMEGA)
            combinedForces += curFor

    return combinedForces * DEPTH


def clip_to_minimum(signal: np.ndarray) -> np.ndarray:
    """
    Detect the minimum (valley) in a 1D signal and return the signal starting
    from that point onward. If no clear minimum is found, return the original signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values.

    Returns
    -------
    np.ndarray
        The clipped signal, starting at the minimum point.
    """
    n = signal.size
    if n < 3:
        return signal.copy()

    min_idx = 0
    min_val = signal[0]

    # Find the global minimum
    for i in range(1, n):
        if signal[i] < min_val:
            min_val = signal[i]
            min_idx = i
        else:
            break
    print(f"I: {i}")
    # Optional safety: ensure it’s not already increasing before start
    if min_idx == 0:
        return signal.copy()

    return signal[min_idx:].copy()


# @njit(cache=True)
def detect_ramp_and_stable(signal, time=None, tolerance=0.03, min_stable_duration=1):
    """
    Detects ramp-up and stable regions in a signal.

    Parameters:
        signal : array-like
            Input signal (e.g., voltage, temperature, etc.)
        time : array-like or None
            Corresponding time array (if None, assumes 1 sample per unit time)
        tolerance : float
            Maximum allowed relative fluctuation (e.g., 0.02 = ±2%) for stability detection
        min_stable_duration : int
            Minimum number of consecutive samples that must be stable to count as "steady"

    Returns:
        ramp_end_idx : int
            Index where ramp-up ends
        stable_start_idx : int
            Index where stability begins
        stable_end_idx : int
            Index where stability ends
    """
    # signal = np.array(signal)
    # minVal = signal.min()
    # maskMin = signal <= minVal
    # signal = clip_to_minimum(signal)

    if time is None:
        time = np.arange(len(signal))

    # Compute gradient (rate of change)
    # grad = np.gradient(signal)
    grad = np.concat([[0], signal[1:] - signal[:-1]])

    # Normalize for relative tolerance
    # avg_signal = np.mean(signal)
    gradStrenght = np.max(np.abs(grad))
    # grad_threshold = tolerance * gradStrenght
    # grad_threshold = 0.001

    # Find where signal stops changing much (below threshold)
    th = signal.mean()
    "DESCENDING & ASCENDING MASK & Signal Top or Bottom"
    topMask = (grad > -gradStrenght * 0.22) & (grad < gradStrenght * 0.25) & (signal > th)
    bottomMask = (grad > -gradStrenght * 0.1) & (grad < gradStrenght * 0.05) & (signal < th)
    # stable_mask = (np.abs(grad) <= grad_threshold) & (signal > th)
    # Find first long-enough stable region
    stable_mask = topMask
    count = 0
    stable_start_idx = None
    for i, is_stable in enumerate(stable_mask):
        if is_stable:
            count += 1
            if count >= min_stable_duration:
                stable_start_idx = i - min_stable_duration + 1
                break
        else:
            count = 0

    if stable_start_idx is None:
        print("No stable region detected. Try increasing tolerance or lowering min_stable_duration.")
        # plt.close("all")
        # plt.title("Not stable!")
        # plt.plot(time, signal, label="Signal", lw=2)
        # plt.show()
        return -0.1, -0.1, -0.1

    # Define ramp-up as everything before stable region
    ramp_end_idx = stable_start_idx
    # stable_mask = np.abs(grad) < 0.01

    ramp_start_index = 0
    while ramp_start_index < ramp_end_idx and bottomMask[ramp_start_index]:
        ramp_start_index += 1

    # Find when stability ends
    stable_end_idx = stable_start_idx
    while stable_end_idx < len(stable_mask) and stable_mask[stable_end_idx]:
        stable_end_idx += 1

    decayEnd_index = stable_end_idx
    while (decayEnd_index < len(bottomMask)) and not (bottomMask[decayEnd_index]):
        decayEnd_index += 1

    # Plot result
    # plt.close("all")
    # plt.plot(topMask)
    # plt.plot(bottomMask)
    # plt.figure(figsize=(10, 5))
    # plt.plot(time, signal, label="Signal", lw=2)
    # # plt.plot(time, grad, label="grad")
    # plt.axvline(time[ramp_start_index], color='black', linestyle='--', label="Ramp start")
    # plt.axvline(time[ramp_end_idx], color='orange', linestyle='--', label="Ramp end")
    # plt.axvline(time[stable_start_idx], color='green', linewidth=2, linestyle='--', label="Stable start")
    # plt.axvline(time[stable_end_idx - 1], color='red', linestyle='--', label="Stable end")
    # plt.axvline(time[decayEnd_index - 1], color='blue', linestyle='--', label="Decay End")
    # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("Signal")
    # plt.title("Ramp-up and Stable Region Detection")
    # plt.show()
    rampLen = ramp_end_idx - ramp_start_index
    stableLen = stable_end_idx - stable_start_idx
    decayLen = decayEnd_index - stable_end_idx
    return rampLen / len(signal), stableLen / len(signal), decayLen / len(signal)


@measure_real_time_decorator
def PlotToolContact(FLUTES=2, ENGAGE=0.5, DEPTH=3, DoPlot=True, resolution=1):
    ROTATION = np.arange(-90, 180 * 2 + 30, resolution, dtype=float)
    # DEPTH = np.linspace(2, 6, 300)
    # CHIP = 0.3
    # FLUTES = 2
    # ENGAGE = 0.75
    CYCLE_DEPTH = 10  # Height to make 360
    # DEPTH = 3
    SELECTION_MASK = (180 <= ROTATION) & (ROTATION <= (180 + 360))

    AVG_N = 3
    angleStep = 360 / FLUTES
    endAngle = DEPTH / CYCLE_DEPTH * 360
    usedCmap = get_cmap("gist_rainbow")

    # THRESHOLD = np.sin((ENGAGE) * np.pi) * 90
    # for en in np.linspace(0, 1, 21):
    # print(f"Engage: {en:>3.2f}, alfa: {np.sin((0.5 - en) * np.pi) * 90:>4.2f}")
    # print(f"Threshold: {THRESHOLD} for engage: {ENGAGE}")

    if DoPlot:
        plt.subplots(
            2, 2, figsize=(12, 8),
            sharex=True,
        )
        plt.subplot(2, 2, 1)
    combinedForces = getCombined(ROTATION, FLUTES, DEPTH, CYCLE_DEPTH, ENGAGE=ENGAGE)
    if AVG_N > 0:
        # combinedForces = moving_average(combinedForces, AVG_N)
        combinedForces = moving_average(combinedForces, AVG_N)
    SCOPE = combinedForces[SELECTION_MASK]
    MAX_FORCE = SCOPE.max()
    GAP = detect_ramp_and_stable(SCOPE)
    DIFF_CLIMB = SCOPE.max() - SCOPE.min()
    if DoPlot:
        plt.plot(ROTATION, combinedForces)

    if DoPlot:
        plt.subplot(2, 2, 3)
        for flut in range(FLUTES):
            "INITIAL CUT"
            curFor = CalcEngageAreaAngle(ROTATION, flut * angleStep, engage=ENGAGE)
            plt.plot(ROTATION, curFor, color=usedCmap(flut / FLUTES))

            "END CUT"
            curFor = CalcEngageAreaAngle(ROTATION, flut * angleStep + endAngle, engage=ENGAGE)
            plt.plot(ROTATION, curFor, color=usedCmap(flut / FLUTES), linestyle="dashed")

        plt.subplot(2, 2, 2)
    combinedForces = getCombined(ROTATION, FLUTES, DEPTH, CYCLE_DEPTH, ENGAGE=ENGAGE, useClimb=False)
    if AVG_N > 0:
        combinedForces = moving_average(combinedForces, AVG_N, )
    DIFF_KONW = combinedForces[(180 <= ROTATION) & (ROTATION <= (180 + 360))]
    DIFF_KONW = DIFF_KONW.max() - DIFF_KONW.min()

    if DoPlot:
        plt.plot(ROTATION, combinedForces)

        plt.subplot(2, 2, 4)
        for flut in range(FLUTES):
            "INITIAL CUT"
            curFor = CalcEngageAreaAngle(ROTATION, flut * angleStep, engage=ENGAGE, useClimb=False)
            plt.plot(ROTATION, curFor, color=usedCmap(flut / FLUTES))

            "END CUT"
            curFor = CalcEngageAreaAngle(ROTATION, flut * angleStep + endAngle,
                                         engage=ENGAGE, useClimb=False)
            plt.plot(ROTATION, curFor, color=usedCmap(flut / FLUTES), linestyle="dashed")

    def format():
        plt.grid(True)
        plt.grid(True)

    if DoPlot:
        plt.subplot(2, 2, 1)
        plt.title(f"Integrated cut. Climb. Diff: {DIFF_CLIMB:>3.3f}")
        format()
        plt.ylabel("Contact [mm]")

        plt.subplot(2, 2, 3)
        format()
        plt.title(f"Single cut. Climb")
        plt.ylabel("Contact [mm]")

        plt.subplot(2, 2, 2)
        format()
        plt.title(f"Integrated cut. Conventional. Diff: {DIFF_KONW:>3.3f}")
        plt.subplot(2, 2, 4)
        format()
        plt.title(f"Single cut. Conventional")
        custom_lines = [
            Line2D([0], [0], color='red', lw=2, linestyle='-', label='First contact'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Last contact'),
        ]
        plt.suptitle(
            f"{FLUTES} blades, Engage: {ENGAGE*100:>4.1f}%, Depth: {DEPTH}mm of {CYCLE_DEPTH}mm")
        plt.xlabel("Degrees")
        plt.xticks(np.arange(0, ROTATION[-1] + 5, 90), rotation=30,)
        plt.legend(handles=custom_lines, loc='upper right')

        plt.subplot(2, 2, 3)
        plt.xlabel("Degrees")
        plt.xticks(np.arange(0, ROTATION[-1] + 5, 90), rotation=30,)
        plt.legend(handles=custom_lines, loc='upper right')

        plt.tight_layout()

    return DIFF_CLIMB, MAX_FORCE, GAP


def PlotMesh(FLUTES=2, ENGAGE=0.2, useClimb=True):
    # FLUTES = 3
    ROTATION = np.linspace(-5, 180 * 2 + 30, 1000, dtype=float)
    DEPTH = np.linspace(2, 10, 10, dtype=float)
    ROTATION = np.arange(-15, 180 * 2 + 5, 1, dtype=float)
    DEPTH = np.arange(2, 10 + .01, 0.5, dtype=float)
    ROTATION = np.arange(-15, 180 * 2 + 5, 5, dtype=float)
    DEPTH = np.arange(2, 10 + .01, 1, dtype=float)

    # ENGAGE = 0.2
    MAX_DEPTH = 10
    XX, YY = np.meshgrid(ROTATION, DEPTH)
    RESULT = XX * 0

    for ri, rad in enumerate(ROTATION):
        print(f"{rad / ROTATION[-1]:>2.2f}", rad)
        for di, dep in enumerate(DEPTH):
            res = getCombined(
                ROTATION, FLUTES, dep, MAX_DEPTH, ENGAGE, useClimb=useClimb, resolution=1,
                # chip=0.3, radius=4
            )
            res = moving_average(res, 5)
            RESULT[di] = res

    plt.rcParams.update({
        # 'figure.facecolor': 'lightgray',
        # 'axes.facecolor': 'gray'
    })

    plt.contour(XX, YY, RESULT, levels=40, cmap="turbo")
    plt.grid(True)
    plt.colorbar()
    plt.xlabel("Obrót, stopnie")
    plt.ylabel("Głębokość")
    text = "Climb" if useClimb else "Konwencional"
    plt.title(f"Wykres kontaktu ({text}). Ostrza: {FLUTES}, Stopień kontaktu: {ENGAGE*100:>2.1f}%")
    plt.tight_layout()


def test():
    rotation = np.linspace(0, 180 * 3, 1000)
    TH_BIG = 0.8
    ENGAGE = [0.1, 0.3, 0.45, 0.5, 0.65, 0.8]
    plt.figure(figsize=(12, 8))
    for bi, bl in enumerate([True, False]):
        for ei, eng in enumerate(ENGAGE):
            plt.subplot(len(ENGAGE), 2, ei * 2 + bi + 1)
            area = CalcEngageAreaAngle(rotation, 0, eng, bl)
            plt.plot(rotation, area)
            text = "Climb" if bl else "Konwo"
            plt.title(f"{text}, {eng*100:>2.1f}%")
            plt.xticks(np.arange(0, rotation[-1] + 5, 90), rotation=30,)

    plt.tight_layout()


def renderPlots():
    for bl in [True]:
        for flut in [2, 3, 4]:
            for eng in [0.25, 0.33, 0.4, 0.5, 0.66, 0.75]:
                plt.figure(figsize=(10, 7), dpi=200)
                PlotMesh(flut, eng, bl)
                tech = "Climb" if bl else "Konw"
                name = os.path.join("images", f"Contact_{flut}_{tech}_{eng:<2.02f}%.png")
                plt.savefig(name)
                print(f"Saved: {name}")
                plt.close()


def compare():
    ENGAGE = np.arange(0.1, 1.01, 0.025)
    DEPTH = np.arange(0, 8.01, 0.2)
    # ENGAGE = np.arange(0.1, 1.01, 0.05)
    # DEPTH = np.arange(0, 8.01, 0.25)
    # ENGAGE = np.arange(0.1, 1.01, 0.25)
    # DEPTH = np.arange(0, 8.01, 1)

    XX, YY = np.meshgrid(ENGAGE, DEPTH)
    RESDIFF = XX * 0
    RESFORCE = XX * 0
    STABLE = XX * 0
    RAMPIN = XX * 0
    VIBR = XX * 0
    ContactSum = XX * 0

    for fl in [2, 3, 4]:
        plt.figure(figsize=(12, 7), dpi=150)
        for di, dep in enumerate(DEPTH):
            for ei, eng in enumerate(ENGAGE):
                diffClimb, forceCLimb, hillSize = PlotToolContact(fl, eng, dep, False, resolution=1)
                RESDIFF[di, ei] = diffClimb
                RESFORCE[di, ei] = forceCLimb
                RAMPIN[di, ei] = hillSize[0]
                STABLE[di, ei] = hillSize[1]
                VIBR[di, ei] = diffClimb * forceCLimb
                ContactSum[di, ei] = np.sum(hillSize)

        plt.subplot(1, 2, 1)
        plt.contourf(XX, YY, RESDIFF, cmap="turbo", levels=15)
        plt.ylabel("Depth")
        plt.xlabel("Engage")
        plt.grid(True)
        plt.title(f"Pseudo wibracje (różnica oporu), Ostrza: {fl}")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.contourf(XX, YY, RESFORCE, cmap="turbo", levels=15)
        # plt.ylabel("Depth")
        plt.xlabel("Engage")
        plt.grid(True)
        plt.title(f"Opór styku frezu helixowego, Ostrza: {fl}")
        plt.colorbar()

        plt.tight_layout()
        name = f"PseudoWibracje_{fl}_.png"
        plt.savefig(os.path.join("images", name))
        print(f"Saved image: {name}")
        plt.close()

        plt.figure(figsize=(12, 7), dpi=150)
        plt.contourf(XX, YY, STABLE, cmap="terrain_r", levels=15)
        plt.xlabel("Engage")
        plt.ylabel("Depth")
        plt.grid(True)
        plt.colorbar()
        plt.title(f"Dlugosc Stabilnego oporu. Ostrza: {fl}")
        plt.tight_layout()
        name = os.path.join("images", f"Stabilizacja_{fl}_.png")
        # plt.savefig(name)
        plt.close()

        plt.figure(figsize=(12, 7), dpi=150)
        plt.contourf(XX, YY, VIBR, cmap="turbo", levels=15)
        plt.xlabel("Engage")
        plt.ylabel("Depth")
        plt.grid(True)
        plt.colorbar()
        plt.title(f"Pseudo wibracje skalowane z oporem, Ostrza: {fl}")
        contours = plt.contour(XX, YY, VIBR, levels=15, linewidths=1.5, colors="black")

        # Add labels on contour lines
        plt.clabel(contours, inline=True, fontsize=8)

        plt.tight_layout()
        name = f"PseudoWibracje_{fl}_Skalowane.png"
        name = os.path.join("images", name)
        plt.savefig(name)
        plt.close()

        plt.figure(figsize=(12, 8), dpi=150)
        plt.subplot(2, 1, 1)
        plt.contourf(XX, YY, RAMPIN, cmap="terrain_r", levels=15)
        plt.xlabel("Engage")
        plt.ylabel("Depth")
        plt.grid(True)
        plt.colorbar()
        plt.title(f"Rampa Wejscia oporu. Ostrza: {fl}")

        plt.subplot(2, 1, 2)
        plt.contourf(XX, YY, ContactSum, cmap="terrain_r", levels=15)
        plt.title(f"SUMA Rampy + Stablinosci oporu. Ostrza: {fl}")
        plt.xlabel("Engage")
        plt.ylabel("Depth")
        plt.grid(True)
        plt.colorbar()
        plt.tight_layout()
        name = os.path.join("images", f"Rampy_{fl}_.png")
        # plt.savefig(name)
        plt.close()


if __name__ == "__main__":
    # plt.figure(figsize=(12, 8))
    # PlotToolContact(2, 0.5, 2)
    # PlotToolContact(2, 0.5, 5)
    # PlotToolContact(2, 0.5, 3)
    # PlotToolContact(2, 0.5, 1)
    # PlotToolContact(2, 0.1, 1)
    # PlotToolContact(2, 0.3, 2)
    # PlotToolContact(2, 0.5, 4.8)
    # PlotToolContact(2, 0.3, 2)
    PlotToolContact(3, 0.3, 5)
    # PlotToolContact(2, 1, 2)
    plt.show()
    # renderPlots()
    # compare()
