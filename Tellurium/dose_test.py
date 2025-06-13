import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your Antimony model as a string
with open("Geerts_2023_Antimony-3.txt", "r") as f:
    antimony_str = f.read()

rr = te.loada(antimony_str)

# Set up the dosing parameters for a single SC dose at t=0
rr.SC_DoseAmount =  2050.6        # Dose amount
rr.SC_DoseDuration = 1        # Duration of dose (hours)
rr.SC_NumDoses = 1            # Only one dose
rr.SC_DoseInterval = 1        # Interval (irrelevant for single dose)
rr.MaxDosingTime = 2040       # Make sure dosing is allowed during simulation
# Optionally, ensure InputCent is zero if you want no IV dosing
# rr.InputCent = 0

# Simulate for 2040 hours
result = rr.simulate(0, 2040, 500, selections=['time', 'InputSC', '[PK_p_brain]'])

# Load Geerts_Gant_Data.csv (assume columns: time, PK_p_brain_data)
data = pd.read_csv('Geerts_Gant_Data.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(result[:,0]/24, result[:,1], label='InputSC', color='tab:blue')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('InputSC')
ax1.set_title('SC Input Function')
ax1.legend()

# PK_p_brain simulation as black line
ax2.semilogy(result[:,0]/24, result[:,2], label='PK_p_brain (sim)', color='black', linewidth=2)
# Data as red circles
ax2.semilogy(data['Time'], data['Concentration'], 'ro', label='Data')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('PK_p_brain')
ax2.set_title('PK_p_brain Response')
ax2.set_ylim(0.1, 1000)
ax2.legend()

plt.tight_layout()
plt.show()