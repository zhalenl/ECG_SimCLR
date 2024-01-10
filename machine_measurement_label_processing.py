import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


machine_measurements = pd.read_csv(
    f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv"
)
record_list = pd.read_csv(
    f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv"
)

machine_measurements = pd.concat([machine_measurements, record_list], axis=1)

machine_measurements = machine_measurements.dropna(subset=["report_0"]).reset_index(
    drop=True
)
sinus_rhythm = np.array([0] * len(machine_measurements))
sinus_bradycardia = np.array([0] * len(machine_measurements))
sinus_tachycardia = np.array([0] * len(machine_measurements))
atrial_fibrillation = np.array([0] * len(machine_measurements))
data_warning = np.array([0] * len(machine_measurements))

for i, r in enumerate(machine_measurements["report_0"]):
    if "sinus rhythm" in r.lower():
        sinus_rhythm[i] = 1
    if "sinus bradycardia" in r.lower():
        sinus_bradycardia[i] = 1
    if "sinus tachycardia" in r.lower():
        sinus_tachycardia[i] = 1
    if "atrial fibrillation" in r.lower():
        atrial_fibrillation[i] = 1
    if r == "--- Warning: Data quality may affect interpretation ---":
        data_warning[i] = 1

machine_measurements["sinus_rhythm"] = sinus_rhythm
machine_measurements["sinus_bradycardia"] = sinus_bradycardia
machine_measurements["sinus_tachycardia"] = sinus_tachycardia
machine_measurements["atrial_fibrillation"] = atrial_fibrillation
machine_measurements["data_warning"] = data_warning

# machine_measurements['sum'] = machine_measurements["sinus_rhythm"] + machine_measurements["sinus_bradycardia"] + machine_measurements["sinus_tachycardia"] + machine_measurements["atrial_fibrillation"]

print("before dropping double labels", len(machine_measurements))
machine_measurements = machine_measurements[
    machine_measurements["sinus_rhythm"]
    + machine_measurements["sinus_bradycardia"]
    + machine_measurements["sinus_tachycardia"]
    + machine_measurements["atrial_fibrillation"]
    < 2
].reset_index(drop=True)
print("after dropping double labels", len(machine_measurements))

label = []
valid_bit = []

for i in tqdm(range(len(machine_measurements))):
    ecg1 = np.load(
        # f"{self.meta_data.iloc[idx]['segment_path']}{self.meta_data.iloc[idx]['segment_path'][-9:-1]}_ECG_II.npy"
        f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/{machine_measurements.iloc[i]['path']}_ECG_II.npy"
    )
    ecg2 = np.load(
        # f"{self.meta_data.iloc[idx]['segment_path']}{self.meta_data.iloc[idx]['segment_path'][-9:-1]}_ECG_II.npy"
        f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/{machine_measurements.iloc[i]['path']}_ECG_III.npy"
    )
    if not (np.isfinite(ecg1).all() & np.isfinite(ecg2).all()):
        valid_bit.append(0)
        print("an invalid ecg - having nan values")
    else:
        valid_bit.append(1)

    if machine_measurements.iloc[i]["sinus_rhythm"] == 1:
        label.append("sinus_rhythm")
    elif machine_measurements.iloc[i]["sinus_bradycardia"] == 1:
        label.append("sinus_bradycardia")
    elif machine_measurements.iloc[i]["sinus_tachycardia"] == 1:
        label.append("sinus_tachycardia")
    elif machine_measurements.iloc[i]["atrial_fibrillation"] == 1:
        label.append("atrial_fibrillation")
    else:
        label.append("other")


machine_measurements["abnormality_label"] = label
machine_measurements["valid_bit"] = valid_bit


machine_measurements.to_csv(
    f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/machine_measurements_abnormality_v3.csv",
    index=False,
)
print("before dropping invalid ecgs", len(machine_measurements))
print("after dropping invalid ecgs", sum(machine_measurements["valid_bit"]))
# root_path = f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/"

# ecg_all = np.load(
#     f"{root_path}{machine_measurements.iloc[0]['path']}_ECG_II.npy"
# ).reshape(1, 5000)
# for idx in tqdm(range(1, len(machine_measurements))):
#     ecg = np.load(
#         f"{root_path}{machine_measurements.iloc[idx]['path']}_ECG_II.npy"
#     ).reshape(1, 5000)
#     ecg_all = np.concatenate((ecg_all, ecg), axis=0)

# print(len(ecg_all))
# with open(
#     f"/ssd-shared/mimiciv-ecg-echonotes/physionet.org/files/mimic-iv-ecg/1.0/processed/ecg_all_abnormality_v1.npy",
#     "wb",
# ) as f:
#     np.save(f, ecg_all)
