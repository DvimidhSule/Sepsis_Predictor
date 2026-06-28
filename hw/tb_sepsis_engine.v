// tb_sepsis_engine.v -- Bit-exact co-simulation testbench
// Drives input features with test vectors generated from the Python model
// and asserts that the RTL outputs match Python exactly.
`timescale 1ns/1ps

module tb_sepsis_engine;
    reg  signed [13:0] temp, resp, hr, sbp;
    reg  temp_v, resp_v, hr_v, sbp_v;
    wire signed [15:0] margin;
    wire sepsis_alarm;

    sepsis_engine dut(.temp(temp), .resp(resp), .hr(hr), .sbp(sbp),
                      .temp_v(temp_v), .resp_v(resp_v), .hr_v(hr_v), .sbp_v(sbp_v),
                      .margin(margin), .sepsis_alarm(sepsis_alarm));

    integer fd, r, n, mism;
    integer e_margin, e_alarm;
    reg [1023:0] line;

    initial begin
        $dumpfile("hw/tb_sepsis_engine.vcd");
        $dumpvars(0, tb_sepsis_engine);

        fd = $fopen("hw/golden_vectors.csv", "r");
        if (fd == 0) begin $display("ERROR: cannot open golden_vectors.csv"); $finish; end
        r = $fgets(line, fd);   // skip header
        n = 0; mism = 0;

        while (!$feof(fd)) begin
            r = $fscanf(fd, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                        temp, resp, hr, sbp, temp_v, resp_v, hr_v, sbp_v, e_margin, e_alarm);
            if (r == 10) begin
                #1; // let combinational logic settle
                if (margin !== e_margin[15:0] || sepsis_alarm !== e_alarm[0]) begin
                    mism = mism + 1;
                    if (mism <= 5)
                        $display("MISMATCH row %0d: RTL margin=%0d alarm=%b | PY margin=%0d alarm=%0d",
                                 n, margin, sepsis_alarm, e_margin, e_alarm);
                end
                n = n + 1;
            end
        end
        $fclose(fd);

        $display("================ CO-SIMULATION ================");
        $display("Rows checked : %0d", n);
        $display("Mismatches   : %0d", mism);
        if (mism == 0) $display("RESULT: BIT-EXACT. RTL == Python on all %0d test rows.", n);
        else           $display("RESULT: FAIL -- %0d rows differ.", mism);
        $display("===============================================");
        $finish;
    end
endmodule
