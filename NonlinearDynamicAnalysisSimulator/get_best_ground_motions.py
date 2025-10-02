import os
import eqsig.single
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy import constants


T_start = 0.06  # 0.2 * the smallest 1st mode period, unit: sec
T_end = 2.6     # 2 * the largest 1st mode period, unit: sec
periods = np.linspace(0.001, 5, 100)
target_start, target_end = None, None  # indexes for t = min_t ~ max_t
for i, t in enumerate(periods):
    if target_start == None and t >= T_start:
        target_start = i-1
    if target_end == None and t > T_end:
        target_end = i
print("start:", target_start, "T:", periods[target_start])
print("end:", target_end, "T:", periods[target_end])


# Design spectrum for Taipei Zone III
S_DS = 0.6
T0 = 1.05
design_spectrum_BSE1 = np.zeros(100)
design_spectrum_BSE2 = np.zeros(100)
for i, t in enumerate(periods):
    if t < 0.2 * T0:
        design_spectrum_BSE1[i] = S_DS * (0.4 + 3 * t / T0)
    elif t >= 0.2 * T0 and t <= T0:
        design_spectrum_BSE1[i] = S_DS
    elif t > T0:
        design_spectrum_BSE1[i] = S_DS * T0 / t
    else:
        design_spectrum_BSE1[i] = None
design_spectrum_BSE2 = design_spectrum_BSE1 * (4/3)


# Tony's ground motion selection and scaling
dt = 0.005  # same 200 Hz for all ground motion records
design_spectrum = design_spectrum_BSE2[target_start:target_end]

def _read_ground_motions(ground_motion_dir):
    ground_motion_names = []
    spectrums = []
    for gm_name in tqdm(os.listdir(ground_motion_dir)):
        gm_folder = ground_motion_dir / gm_name

        # there should be a XXX_FN.txt and a XXX_FP.txt
        files = [gm_folder / file for file in os.listdir(gm_folder)]
        assert len(files) == 2, "Both FN and FP files should exist"

        # read gm
        gm_fn = np.loadtxt(files[0])[:, 1] / 1000 / 9.8
        gm_fp = np.loadtxt(files[1])[:, 1] / 1000 / 9.8

        # get spectrums
        record_fn = eqsig.AccSignal(gm_fn, dt)
        record_fp = eqsig.AccSignal(gm_fp, dt)
        record_fn.generate_response_spectrum(response_times=periods)
        record_fp.generate_response_spectrum(response_times=periods)

        # get the spectrum values within target period
        sa_fn = record_fn.s_a[target_start:target_end]
        sa_fp = record_fp.s_a[target_start:target_end]

        # get the geometric mean of the spectrum
        geometric_mean_sa = np.sqrt(sa_fn * sa_fp)
        geometric_mean_sa = geometric_mean_sa[np.newaxis, :]
        
        ground_motion_names.append(gm_name)
        spectrums.append(geometric_mean_sa)

    return ground_motion_names, spectrums


def greedy_select_ground_motion(spectrums, ground_motion_names, target_number=11):
    total_num = len(spectrums)
    selected_gms = []

    while len(selected_gms) < target_number:

        best_gm = None
        best_gm_difference = np.inf

        current_selected_gms = [spectrums[k] for k in selected_gms]

        for i in range(total_num):

            # if gm is already selected than pass
            if i in selected_gms:   continue

            # calculate the average of the current gm and the selected best gms
            current_gm = spectrums[i]
            all_gm = np.concatenate(current_selected_gms + [current_gm], axis=0)
            average_gm = np.mean(all_gm, axis=0)

            # calculate the distance between current gm and target design spectrum
            target_difference = np.sum((average_gm - design_spectrum) ** 2)
            if target_difference < best_gm_difference:
                best_gm_difference = target_difference
                best_gm = i

        # update the new best-fit gm
        selected_gms.append(best_gm)
        print(f"current best gm: {best_gm}, gm_name: {ground_motion_names[best_gm]}")

    print("selected_gms:", selected_gms)
    return selected_gms


def scale_selected_ground_motions(spectrums, selected_gms):
    # First check each spectrum average should greater than the design spectrum,
    # if not, then scale the spectrum.
    selected_spectrums = [spectrums[i] for i in selected_gms]
    scale_factors = []
    for i in range(len(selected_spectrums)):
        spectrum = selected_spectrums[i]
        # print(f"design: {design_spectrum.shape}, spec: {spectrum.shape}")
        scale_factor = np.mean(design_spectrum) / np.mean(spectrum)
        scale_factor = max(1, scale_factor)     # only scale up don't scale down
        selected_spectrums[i] = spectrum * scale_factor
        scale_factors.append(scale_factor)

    # Then calculate the average of the selected spectrums, if any point of the average
    # spectrum is smaller than the design spectrum, all of the ground motion should scale up.
    selected_spectrums = np.concatenate(selected_spectrums, axis=0)
    spectrum_average = np.mean(selected_spectrums, axis=0)

    # plt.plot(design_spectrum)
    # plt.plot(spectrum_average)
    # plt.show()
    overall_scale_factor = np.max(design_spectrum * 0.9 / spectrum_average)
    # print("overall scale_factor:", overall_scale_factor)
    scale_factors = [fac * overall_scale_factor for fac in scale_factors]
    # print("scale factors:", scale_factors)

    return scale_factors


def generate_new_ground_motions(ground_motion_names, selected_gms, scale_factors, target_dir):
    # first load the ground motions, then scale the ground motions and save to new folder
    original_spectrums = []
    scaled_spectrums = []
    for i in tqdm(range(len(selected_gms))):
        gm_index = selected_gms[i]
        gm_name = ground_motion_names[gm_index]
        # print(gm_name + ", ")
        gm_folder = ground_motion_dir / gm_name

        # there should be a XXX_FN.txt and a XXX_FP.txt
        files = [gm_folder / file for file in os.listdir(gm_folder)]
        assert len(files) == 2, "Both FN and FP files should exist"

        # read gm
        gm_fn = np.loadtxt(files[0])
        gm_fp = np.loadtxt(files[1])

        # scale
        gm_fn_new = gm_fn.copy()
        gm_fp_new = gm_fp.copy()
        gm_fn_new[:, 1] *= scale_factors[i]
        gm_fp_new[:, 1] *= scale_factors[i]

        # save it to target folder
        new_gm_name = gm_name + f"_{scale_factors[i]:.4f}"
        new_folder = target_dir / new_gm_name
        new_folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(new_folder / f"{gm_name}_FN.txt", gm_fn_new, delimiter='\t')
        np.savetxt(new_folder / f"{gm_name}_FP.txt", gm_fp_new, delimiter='\t')


        # get the spectrum for visualization
        record_fn = eqsig.AccSignal(gm_fn[:, 1] / 1000 / 9.8, dt)
        record_fp = eqsig.AccSignal(gm_fp[:, 1] / 1000 / 9.8, dt)
        record_fn.generate_response_spectrum(response_times=periods)
        record_fp.generate_response_spectrum(response_times=periods)
        times = record_fp.response_times
        sa_fn = record_fn.s_a
        sa_fp = record_fp.s_a
        geometric_mean_sa = np.sqrt(sa_fn * sa_fp)
        geometric_mean_sa = geometric_mean_sa[np.newaxis, :]
        original_spectrums.append(geometric_mean_sa)

        # spectrum for the scaled ground motions
        record_fn_new = eqsig.AccSignal(gm_fn_new[:, 1] / 1000 / 9.8, dt)
        record_fp_new = eqsig.AccSignal(gm_fp_new[:, 1] / 1000 / 9.8, dt)
        record_fn_new.generate_response_spectrum(response_times=periods)
        record_fp_new.generate_response_spectrum(response_times=periods)
        sa_fn_new = record_fn_new.s_a
        sa_fp_new = record_fp_new.s_a
        geometric_mean_sa_new = np.sqrt(sa_fn_new * sa_fp_new)
        geometric_mean_sa_new = geometric_mean_sa_new[np.newaxis, :]
        scaled_spectrums.append(geometric_mean_sa_new)

    # then plot the original and scaled selected ground motions
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for i in range(len(original_spectrums)):
        axs[0].plot(times, original_spectrums[i].squeeze(), color="silver", linewidth=1)
        axs[1].plot(times, scaled_spectrums[i].squeeze(), color="silver", linewidth=1)
    
    original_spectrums = np.concatenate(original_spectrums, axis=0)
    scaled_spectrums = np.concatenate(scaled_spectrums, axis=0)
    axs[0].plot(times, design_spectrum_BSE2, color="blue", linewidth=2, label="MCE (2%/50y)")
    axs[0].plot(times, np.mean(original_spectrums, axis=0).squeeze(), color="black", linewidth=2, label="Original")

    axs[1].plot(times, design_spectrum_BSE2, color="blue", linewidth=2, label="MCE (2%/50y)")
    axs[1].plot(times, np.mean(scaled_spectrums, axis=0).squeeze(), color="black", linewidth=2, label="Scaled")

    axs[0].legend()
    axs[0].set_xlabel("T (sec)")
    axs[0].set_ylabel("Sa (G)")
    axs[0].set_title(f"The selected ground motions")

    axs[1].legend()
    axs[1].set_xlabel("T (sec)")
    axs[1].set_ylabel("Sa (G)")
    axs[1].set_title(f"The scaled selected ground motions")
    plt.savefig(f"selected_spectrum.png")
    plt.close()


# Jack's ground motion selection and scaling

# INMOST ground motion dataset --> neglect: pulse
# neglected_gm_names = ["Rank001", "Rank012", "Rank015", "Rank016", "Rank025", "Rank054"]

# World_processed ground motion dataset --> neglect: pulse, missing, weird time-series tendency, large MSE
pulse_gm_names = ['EQ161', 'EQ170', 'EQ171', 'EQ173', 'EQ178', 'EQ179', 'EQ180', 'EQ181', 'EQ182', 'EQ184', 'EQ185', 'EQ764', 'EQ766', 'EQ802', 'EQ803', 'EQ1004', 'EQ1084', 'EQ1085', 'EQ1176']
missing_gm_names = ['EQ465', 'EQ466', 'EQ994', 'EQ1009', 'EQ1178']
weird_gm_names = ['EQ450', 'EQ986']
unmatched_gm_names = ['EQ122', 'EQ125', 'EQ162', 'EQ187', 'EQ367', 'EQ461', 'EQ548', 'EQ558', 'EQ728', 'EQ731', 'EQ732', 'EQ735', 'EQ739', 'EQ748', 'EQ751', 'EQ753', 'EQ755', 'EQ759', 'EQ761', 'EQ762', 'EQ768', 'EQ769', 'EQ772', 'EQ776', 'EQ777', 'EQ778', 'EQ791', 'EQ806', 'EQ812', 'EQ4350', 'EQ8063']
print(f"pulse: {len(pulse_gm_names)}, missing: {len(missing_gm_names)}, weird: {len(weird_gm_names)}, unmatched: {len(unmatched_gm_names)}")
neglected_gm_names = pulse_gm_names + missing_gm_names + weird_gm_names
print(f"neglect {len(neglected_gm_names)} pairs of ground motions")
keep_gm_names = ['EQ20', 'EQ175', 'EQ464', 'EQ721', 'EQ729', 'EQ736', 'EQ786', 'EQ787', 'EQ881', 'EQ1628', 'EQ3752', 'EQ3754', 'EQ3757', 'EQ3758', 'EQ3759', 'EQ4013', 'EQ4084', 'EQ4141', 'EQ4144', 'EQ4145', 'EQ4146', 'EQ4149', 'EQ4346', 'EQ4348', 'EQ4481', 'EQ6930', 'EQ6953', 'EQ8067', 'EQ8102', 'EQ8118', 'EQ8134']
total_gm_names = (pulse_gm_names + missing_gm_names + weird_gm_names + unmatched_gm_names) + keep_gm_names

def read_ground_motions(ground_motion_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """Read the ground motions and get the response spectrums in the given directory."""
    ground_motion_names = []
    spectrums = []
    for gm_name in tqdm(os.listdir(ground_motion_dir)):
        if gm_name in neglected_gm_names: continue  # skip the pulse ground motions

        # there should be a XXX_FN.txt and a XXX_FP.txt
        gm_folder = ground_motion_dir / gm_name
        files = [gm_folder / file for file in os.listdir(gm_folder)]
        assert len(files) == 2, "Both FN and FP files should exist"

        # read gm (unit: mm/s^2 --> g)
        gm_fn = np.loadtxt(files[0])[:, 1] / 1000 / constants.g  # 9.80665 m/s^2
        gm_fp = np.loadtxt(files[1])[:, 1] / 1000 / constants.g  # 9.80665 m/s^2
        time_series = np.loadtxt(files[0])[:, 0]
        dts = np.round(time_series[1:] - time_series[:-1], 3)  # round the values to avoid precision error
        assert np.all(dts == dts[0]), "Time series should be evenly spaced"
        dt = dts[0]
        print(gm_name, dt)

        # get spectrums
        record_fn = eqsig.AccSignal(gm_fn, dt)
        record_fp = eqsig.AccSignal(gm_fp, dt)
        record_fn.generate_response_spectrum(response_times=periods)
        record_fp.generate_response_spectrum(response_times=periods)

        # get the geometric mean of the spectrum acceleration
        sa_fn = record_fn.s_a
        sa_fp = record_fp.s_a
        geometric_mean_sa = np.sqrt(sa_fn * sa_fp)
        
        ground_motion_names.append(gm_name)
        spectrums.append(geometric_mean_sa)

    return ground_motion_names, spectrums


def scale_each_ground_motion(orignal_spectrums: list[np.ndarray]) -> tuple[list[float], list[np.ndarray]]:
    """Scale each ground motion spectrum to the target design spectrum."""
    scaled_factors = []
    scaled_spectrums = []
    for i, original_spectrum in enumerate(orignal_spectrums):
        scale_factor = np.mean(design_spectrum_BSE2[target_start:target_end]) / np.mean(original_spectrum[target_start:target_end])
        scale_factor = max(1, scale_factor)  # only scale up
        scaled_spectrum = original_spectrum * scale_factor
        scaled_factors.append(scale_factor)
        scaled_spectrums.append(np.expand_dims(scaled_spectrum, axis=0))  # shape: (1, 100) --> ready for concatenation at axis=0
    
    return scaled_factors, scaled_spectrums


def greedy_select_ground_motions(scaled_spectrums: list[np.ndarray], target_number=11) -> list[int]:
    """Select the best combination of ground motions by greedy search."""
    all_selected_gm_indexs = []
    all_best_gm_differences = []
    for first_index in range(len(scaled_spectrums)):
        # use each ground motion as the first selected ground motion
        selected_gm_indexs = [first_index]
        while len(selected_gm_indexs) < target_number:
            best_gm_index = None
            best_gm_difference = np.inf
            current_selected_gms = [scaled_spectrums[k] for k in selected_gm_indexs]

            for i in range(len(scaled_spectrums)):
                # if gm is already selected than pass
                if i in selected_gm_indexs: continue

                # calculate the average of the current gm and the selected best gms
                current_gm = scaled_spectrums[i]
                all_gm = np.concatenate(current_selected_gms + [current_gm], axis=0)
                average_gm = np.mean(all_gm, axis=0)

                # calculate the distance between current gm and target design spectrum
                target_difference = np.sum((average_gm[target_start:target_end] - design_spectrum_BSE2[target_start:target_end]) ** 2)
                if target_difference < best_gm_difference:
                    best_gm_difference = target_difference
                    best_gm_index = i

            selected_gm_indexs.append(best_gm_index)
            # print(f"current best gm: {best_gm_index}, gm_name: {ground_motion_names[best_gm_index]}, MSE: {best_gm_difference}")
        
        all_selected_gm_indexs.append(selected_gm_indexs)
        all_best_gm_differences.append(best_gm_difference)
    
    best_index = np.argmin(all_best_gm_differences)
    selected_gm_indexs = all_selected_gm_indexs[best_index]
    print(f"Best selected gms: {selected_gm_indexs}, MSE: {all_best_gm_differences[best_index]}")

    return selected_gm_indexs


def generate_scaled_ground_motions(selected_scaled_spectrums: list[np.ndarray], selected_scaled_factors: list[float], selected_ground_motion_names: list[str], target_dir: Path) -> None:
    """Generate the new ground motions by scaling factors calculated from the selected spectrums and save them to the target folder."""
    # overall scale the selected spectrums to the target spectrum
    selected_scaled_spectrums_average = np.concatenate(selected_scaled_spectrums, axis=0).mean(axis=0)
    overall_scaled_factor = np.max(design_spectrum_BSE2[target_start:target_end] * 0.9 / selected_scaled_spectrums_average[target_start:target_end])
    print(overall_scaled_factor)
    overall_selected_scaled_factors = [factor * overall_scaled_factor for factor in selected_scaled_factors]
    print(overall_selected_scaled_factors)

    original_selected_spectrums = []
    scaled_selected_spectrums = []
    for i, selected_ground_motion_name in enumerate(selected_ground_motion_names):
        gm_folder = original_gm_dir / selected_ground_motion_name
        files = [gm_folder / file for file in os.listdir(gm_folder)]
        assert len(files) == 2, "Both FN and FP files should exist"

        # read the ground motion
        gm_fn = np.loadtxt(files[0])
        gm_fp = np.loadtxt(files[1])
        time_series = gm_fn[:, 0]
        dts = np.round(time_series[1:] - time_series[:-1], 3)  # round the values to avoid precision error
        assert np.all(dts == dts[0]), "Time series should be evenly spaced"
        dt = dts[0]

        # scale the ground motion
        gm_fn_new = gm_fn.copy()
        gm_fp_new = gm_fp.copy()
        gm_fn_new[:, 1] *= overall_selected_scaled_factors[i]
        gm_fp_new[:, 1] *= overall_selected_scaled_factors[i]

        # save it to target folder
        new_gm_name = selected_ground_motion_name + f"_{overall_selected_scaled_factors[i]:.4f}"
        new_folder = target_dir / new_gm_name
        new_folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(new_folder / f"{selected_ground_motion_name}_FN.txt", gm_fn_new, fmt="%.4f")
        np.savetxt(new_folder / f"{selected_ground_motion_name}_FP.txt", gm_fp_new, fmt="%.4f")

        # spectrum for the original ground motions
        record_fn = eqsig.AccSignal(gm_fn[:, 1] / 1000 / constants.g, dt)  # unit: mm/s^2 --> g
        record_fp = eqsig.AccSignal(gm_fp[:, 1] / 1000 / constants.g, dt)  # unit: mm/s^2 --> g
        record_fn.generate_response_spectrum(response_times=periods)
        record_fp.generate_response_spectrum(response_times=periods)
        sa_fn = record_fn.s_a
        sa_fp = record_fp.s_a
        geometric_mean_sa = np.sqrt(sa_fn * sa_fp)
        geometric_mean_sa = geometric_mean_sa[np.newaxis, :]
        original_selected_spectrums.append(geometric_mean_sa)

        # spectrum for the scaled ground motions
        record_fn_new = eqsig.AccSignal(gm_fn_new[:, 1] / 1000 / constants.g, dt)  # unit: mm/s^2 --> g
        record_fp_new = eqsig.AccSignal(gm_fp_new[:, 1] / 1000 / constants.g, dt)  # unit: mm/s^2 --> g
        record_fn_new.generate_response_spectrum(response_times=periods)
        record_fp_new.generate_response_spectrum(response_times=periods)
        sa_fn_new = record_fn_new.s_a
        sa_fp_new = record_fp_new.s_a
        geometric_mean_sa_new = np.sqrt(sa_fn_new * sa_fp_new)
        geometric_mean_sa_new = geometric_mean_sa_new[np.newaxis, :]
        scaled_selected_spectrums.append(geometric_mean_sa_new)
    
    # plot the original and scaled selected ground motions
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for i in range(len(original_selected_spectrums)):
        axs[0].plot(periods, original_selected_spectrums[i].squeeze(), color="silver", linewidth=1, label=f"{len(selected_gm_indexs)} selected original spectrums" if i == 0 else None)
        axs[1].plot(periods, scaled_selected_spectrums[i].squeeze(), color="silver", linewidth=1, label=f"{len(selected_gm_indexs)} selected scaled spectrums" if i == 0 else None)

    original_selected_spectrums_average = np.concatenate(original_selected_spectrums, axis=0).mean(axis=0)
    axs[0].plot(periods, original_selected_spectrums_average, color="black", linewidth=2, label=f"average of {len(selected_gm_indexs)} selected original spectrums")
    axs[0].plot(periods, design_spectrum_BSE2, color="blue", linewidth=2, label="MCE-level (2%/50y) design spectrum")
    axs[0].vlines(T_start, 0, 1.5, colors="black", linestyles="dashed", label=f"lower bound of period range: {T_start} sec")
    axs[0].vlines(T_end, 0, 1.5, colors="black", linestyles="dashed", label=f"upper bound of period range: {T_end} sec")
    axs[0].set_xlim(0, 5.1)
    axs[0].set_ylim(0, 1.5)
    axs[0].grid()
    axs[0].legend(loc="best", fontsize=14)
    axs[0].set_xlabel("T (sec)", fontsize=14)
    axs[0].set_ylabel("Sa (g)", fontsize=14)
    axs[0].set_title(f"Original Selected Spectrums & Target Design Spectrum", fontsize=16)

    scaled_selected_spectrums_average = np.concatenate(scaled_selected_spectrums, axis=0).mean(axis=0)
    axs[1].plot(periods, scaled_selected_spectrums_average, color="black", linewidth=2, label=f"average of {len(selected_gm_indexs)} selected scaled spectrums")
    axs[1].plot(periods, design_spectrum_BSE2, color="blue", linewidth=2, label="MCE-level (2%/50y) design spectrum")
    axs[1].vlines(T_start, 0, 1.5, colors="black", linestyles="dashed", label=f"lower bound of period range: {T_start} sec")
    axs[1].vlines(T_end, 0, 1.5, colors="black", linestyles="dashed", label=f"upper bound of period range: {T_end} sec")
    axs[1].set_xlim(0, 5.1)
    axs[1].set_ylim(0, 1.5)
    axs[1].grid()
    axs[1].legend(loc="best", fontsize=14)
    axs[1].set_xlabel("T (sec)", fontsize=14)
    axs[1].set_ylabel("Sa (g)", fontsize=14)
    axs[1].set_title(f"Scaled Selected Spectrums & Target Ddesign Spectrum", fontsize=16)
    plt.savefig(f"selected_spectrum.png")




if __name__ == "__main__":
    ground_motion_dir = Path("./ground_motions")
    original_gm_dir = ground_motion_dir / "GroundMotions_World_Processed_BSE-2"
    scaled_selected_gm_dir = ground_motion_dir / "selected_ground_motions_World_Processed_MCE"
    
    # Tony's ground motion selection and scaling
    # ground_motion_names, spectrums = _read_ground_motions(original_gm_dir)
    # selected_gms = greedy_select_ground_motion(spectrums, ground_motion_names, target_number=11)
    # scale_factors = scale_selected_ground_motions(spectrums, selected_gms)
    # generate_new_ground_motions(ground_motion_names, selected_gms, scale_factors, scaled_selected_gm_dir)

    # Jack's ground motion selection and scaling
    ground_motion_names, original_spectrums = read_ground_motions(original_gm_dir)
    scaled_factors, scaled_spectrums = scale_each_ground_motion(original_spectrums)
    selected_gm_indexs = greedy_select_ground_motions(scaled_spectrums, 11)

    selected_ground_motion_names = [ground_motion_names[i] for i in selected_gm_indexs]
    selected_original_spectrums = [original_spectrums[i] for i in selected_gm_indexs]
    selected_scaled_factors = [scaled_factors[i] for i in selected_gm_indexs]
    selected_scaled_spectrums = [scaled_spectrums[i] for i in selected_gm_indexs]
    print(selected_ground_motion_names)
    print(selected_scaled_factors)
    generate_scaled_ground_motions(selected_scaled_spectrums, selected_scaled_factors, selected_ground_motion_names, scaled_selected_gm_dir)
