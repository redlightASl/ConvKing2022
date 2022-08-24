-- Created by IP Generator (Version 2020.3 build 62942)
-- Instantiation Template
--
-- Insert the following codes into your VHDL file.
--   * Change the_instance_name to your own instance name.
--   * Change the net names in the port map.


COMPONENT JtagCore
  PORT (
    resetn_i : IN STD_LOGIC;
    drck_o : OUT STD_LOGIC;
    hub_tdi : OUT STD_LOGIC;
    capt_o : OUT STD_LOGIC;
    shift_o : OUT STD_LOGIC;
    conf_sel : OUT STD_LOGIC_VECTOR(14 DOWNTO 0);
    id_o : OUT STD_LOGIC_VECTOR(4 DOWNTO 0);
    hub_tdo : IN STD_LOGIC_VECTOR(14 DOWNTO 0)
  );
END COMPONENT;


the_instance_name : JtagCore
  PORT MAP (
    resetn_i => resetn_i,
    drck_o => drck_o,
    hub_tdi => hub_tdi,
    capt_o => capt_o,
    shift_o => shift_o,
    conf_sel => conf_sel,
    id_o => id_o,
    hub_tdo => hub_tdo
  );
