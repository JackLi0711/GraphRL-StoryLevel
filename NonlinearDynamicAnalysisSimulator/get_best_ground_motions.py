import os
import eqsig.single
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


T_start = 0.06  # sec
T_end = 2.6     # sec
dt = 0.005      # 200 Hz
periods = np.linspace(0.01, 10, 100)

target_start, target_end = None, None    # indexes for t = min_t ~ max_t
for i, t in enumerate(periods):
    if target_start == None and t >= T_start:
        target_start = i
    if target_end == None and t > T_end:
        target_end = i

print("start:", target_start)
print("end:", target_end)


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




if __name__ == "__main__":
    ground_motion_dir = Path("ground_motions/GroundMotions_World_processed_BSE-2/")
    target_selected_ground_motion_dir = Path("ground_motions/selected_ground_motions/")
    ground_motion_names, spectrums = _read_ground_motions(ground_motion_dir)
    selected_gms = greedy_select_ground_motion(spectrums, ground_motion_names, target_number=11)
    scale_factors = scale_selected_ground_motions(spectrums, selected_gms)
    generate_new_ground_motions(ground_motion_names, selected_gms, scale_factors, target_selected_ground_motion_dir)