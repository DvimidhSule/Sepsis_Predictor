# tb_demo_patients.tcl -- Configure waves and run simulation in Vivado XSim GUI
# Set up waveforms for the patient demo simulation.

# Add simulation time hour
add_wave -name "Hour" -radix dec /tb_demo_patients/hour

# Group and add bedside decoded vital signs (human readable)
set g_decoded [add_wave_group "Decoded_Vitals"]
add_wave -into $g_decoded -name "Heart_Rate_bpm" -radix dec /tb_demo_patients/HR_bpm
add_wave -into $g_decoded -name "Temperature_C" -radix dec /tb_demo_patients/Temp_C
add_wave -into $g_decoded -name "Resp_Rate_rpm" -radix dec /tb_demo_patients/Resp_rpm
add_wave -into $g_decoded -name "Systolic_BP_mmHg" -radix dec /tb_demo_patients/SBP_mmHg

# Group and add model outputs (decision boundary)
set g_outputs [add_wave_group "Decision_Outputs"]
add_wave -into $g_outputs -name "Model_Margin" -radix dec /tb_demo_patients/margin
add_wave -into $g_outputs -name "Sepsis_Alarm" -radix bin /tb_demo_patients/sepsis_alarm

# Group and add quantized inputs (Q10.4 fixed point)
set g_quantized [add_wave_group "Quantized_Inputs_Q10.4"]
add_wave -into $g_quantized -name "hr_quant" -radix dec /tb_demo_patients/hr
add_wave -into $g_quantized -name "temp_quant" -radix dec /tb_demo_patients/temp
add_wave -into $g_quantized -name "resp_quant" -radix dec /tb_demo_patients/resp
add_wave -into $g_quantized -name "sbp_quant" -radix dec /tb_demo_patients/sbp

# Run the simulation to completion
run all
