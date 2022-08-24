// Created by IP Generator (Version 2020.3 build 62942)
// Instantiation Template
//
// Insert the following codes into your Verilog file.
//   * Change the_instance_name to your own instance name.
//   * Change the signal names in the port associations


JtagCore the_instance_name (
  .resetn_i(resetn_i),    // input
  .drck_o(drck_o),        // output
  .hub_tdi(hub_tdi),      // output
  .capt_o(capt_o),        // output
  .shift_o(shift_o),      // output
  .conf_sel(conf_sel),    // output [14:0]
  .id_o(id_o),            // output [4:0]
  .hub_tdo(hub_tdo)       // input [14:0]
);
