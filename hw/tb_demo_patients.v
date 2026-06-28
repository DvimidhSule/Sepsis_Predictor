// tb_demo_patients.v -- Simulation testbench streaming virtual patients
// Features are 3-hour mean vitals scaled to Q10.4 fixed-point.
// Generates VCD waveform data to analyze alarm timing.
`timescale 1ns/1ps

module tb_demo_patients;
    reg signed [13:0] temp, resp, hr, sbp;
    wire signed [15:0] margin;
    wire sepsis_alarm;

    sepsis_engine dut(.temp(temp), .resp(resp), .hr(hr), .sbp(sbp),
                      .temp_v(1'b1), .resp_v(1'b1), .hr_v(1'b1), .sbp_v(1'b1),
                      .margin(margin), .sepsis_alarm(sepsis_alarm));

    // human-readable vitals for the waveform (undo the x16 fixed-point)
    wire [13:0] HR_bpm   = hr   / 16;
    wire [13:0] Temp_C   = temp / 16;
    wire [13:0] Resp_rpm = resp / 16;
    wire [13:0] SBP_mmHg = sbp  / 16;

    integer hour;
    // helper: apply one hour of vitals (real units), settle, log
    task step(input integer t_c10, input integer rr, input integer bpm, input integer sys);
        begin
            temp = t_c10*16/10; resp = rr*16; hr = bpm*16; sbp = sys*16;
            #5;
            $display("  h%0d  Temp=%0d.%0d Resp=%0d HR=%0d SBP=%0d  -> margin=%0d  ALARM=%b",
                     hour, t_c10/10, t_c10%10, rr, bpm, sys, margin, sepsis_alarm);
            hour = hour + 1; #5;
        end
    endtask

    initial begin
        $dumpfile("tb_demo_patients.vcd");
        $dumpvars(0, tb_demo_patients);

        // ---- Patient A: HEALTHY (stable normal vitals) ----
        $display("=== Patient A: HEALTHY ==="); hour = 0;
        step(368, 16,  72, 122);
        step(369, 15,  74, 120);
        step(370, 16,  73, 121);
        step(368, 17,  75, 119);

        // ---- Patient B: DETERIORATING (slow slide into sepsis) ----
        $display("=== Patient B: DETERIORATING ==="); hour = 0;
        step(370, 17,  78, 120);
        step(376, 19,  92, 112);
        step(382, 22, 104, 100);
        step(386, 25, 116,  92);
        step(389, 27, 124,  86);

        // ---- Patient C: CRASHING (septic shock, fast) ----
        $display("=== Patient C: CRASHING ==="); hour = 0;
        step(390, 28, 130,  82);
        step(393, 30, 138,  78);

        $display("Done. VCD: hw/tb_demo_patients.vcd");
        $finish;
    end
endmodule
