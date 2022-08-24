-- Created by IP Generator (Version 2020.3 build 62942)
-- Instantiation Template
--
-- Insert the following codes into your VHDL file.
--   * Change the_instance_name to your own instance name.
--   * Change the net names in the port map.


COMPONENT DebugCore
  PORT (
    hub_tdi : IN STD_LOGIC;
    hub_tdo : OUT STD_LOGIC;
    id_i : IN STD_LOGIC_VECTOR(4 DOWNTO 0);
    capt_i : IN STD_LOGIC;
    shift_i : IN STD_LOGIC;
    conf_sel : IN STD_LOGIC;
    drck_in : IN STD_LOGIC;
    clk : IN STD_LOGIC;
    resetn_i : IN STD_LOGIC;
    trig0_i : IN STD_LOGIC
  );
END COMPONENT;


the_instance_name : DebugCore
  PORT MAP (
    hub_tdi => hub_tdi,
    hub_tdo => hub_tdo,
    id_i => id_i,
    capt_i => capt_i,
    shift_i => shift_i,
    conf_sel => conf_sel,
    drck_in => drck_in,
    clk => clk,
    resetn_i => resetn_i,
    trig0_i => trig0_i
  );
