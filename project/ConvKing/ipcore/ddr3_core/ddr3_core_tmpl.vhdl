-- Created by IP Generator (Version 2020.3 build 62942)
-- Instantiation Template
--
-- Insert the following codes into your VHDL file.
--   * Change the_instance_name to your own instance name.
--   * Change the net names in the port map.


COMPONENT ddr3_core
  PORT (
    pll_refclk_in : IN STD_LOGIC;
    top_rst_n : IN STD_LOGIC;
    ddrc_rst : IN STD_LOGIC;
    csysreq_ddrc : IN STD_LOGIC;
    csysack_ddrc : OUT STD_LOGIC;
    cactive_ddrc : OUT STD_LOGIC;
    pll_lock : OUT STD_LOGIC;
    pll_aclk_0 : OUT STD_LOGIC;
    pll_aclk_1 : OUT STD_LOGIC;
    pll_aclk_2 : OUT STD_LOGIC;
    ddrphy_rst_done : OUT STD_LOGIC;
    ddrc_init_done : OUT STD_LOGIC;
    pad_loop_in : IN STD_LOGIC;
    pad_loop_in_h : IN STD_LOGIC;
    pad_rstn_ch0 : OUT STD_LOGIC;
    pad_ddr_clk_w : OUT STD_LOGIC;
    pad_ddr_clkn_w : OUT STD_LOGIC;
    pad_csn_ch0 : OUT STD_LOGIC;
    pad_addr_ch0 : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    pad_dq_ch0 : INOUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    pad_dqs_ch0 : INOUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    pad_dqsn_ch0 : INOUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    pad_dm_rdqs_ch0 : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    pad_cke_ch0 : OUT STD_LOGIC;
    pad_odt_ch0 : OUT STD_LOGIC;
    pad_rasn_ch0 : OUT STD_LOGIC;
    pad_casn_ch0 : OUT STD_LOGIC;
    pad_wen_ch0 : OUT STD_LOGIC;
    pad_ba_ch0 : OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    pad_loop_out : OUT STD_LOGIC;
    pad_loop_out_h : OUT STD_LOGIC;
    areset_1 : IN STD_LOGIC;
    aclk_1 : IN STD_LOGIC;
    awid_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    awaddr_1 : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    awlen_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    awsize_1 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
    awburst_1 : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    awlock_1 : IN STD_LOGIC;
    awvalid_1 : IN STD_LOGIC;
    awready_1 : OUT STD_LOGIC;
    awurgent_1 : IN STD_LOGIC;
    awpoison_1 : IN STD_LOGIC;
    wdata_1 : IN STD_LOGIC_VECTOR(63 DOWNTO 0);
    wstrb_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    wlast_1 : IN STD_LOGIC;
    wvalid_1 : IN STD_LOGIC;
    wready_1 : OUT STD_LOGIC;
    bid_1 : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    bresp_1 : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    bvalid_1 : OUT STD_LOGIC;
    bready_1 : IN STD_LOGIC;
    arid_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    araddr_1 : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    arlen_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    arsize_1 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
    arburst_1 : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    arlock_1 : IN STD_LOGIC;
    arvalid_1 : IN STD_LOGIC;
    arready_1 : OUT STD_LOGIC;
    arpoison_1 : IN STD_LOGIC;
    rid_1 : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    rdata_1 : OUT STD_LOGIC_VECTOR(63 DOWNTO 0);
    rresp_1 : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    rlast_1 : OUT STD_LOGIC;
    rvalid_1 : OUT STD_LOGIC;
    rready_1 : IN STD_LOGIC;
    arurgent_1 : IN STD_LOGIC;
    csysreq_1 : IN STD_LOGIC;
    csysack_1 : OUT STD_LOGIC;
    cactive_1 : OUT STD_LOGIC
  );
END COMPONENT;


the_instance_name : ddr3_core
  PORT MAP (
    pll_refclk_in => pll_refclk_in,
    top_rst_n => top_rst_n,
    ddrc_rst => ddrc_rst,
    csysreq_ddrc => csysreq_ddrc,
    csysack_ddrc => csysack_ddrc,
    cactive_ddrc => cactive_ddrc,
    pll_lock => pll_lock,
    pll_aclk_0 => pll_aclk_0,
    pll_aclk_1 => pll_aclk_1,
    pll_aclk_2 => pll_aclk_2,
    ddrphy_rst_done => ddrphy_rst_done,
    ddrc_init_done => ddrc_init_done,
    pad_loop_in => pad_loop_in,
    pad_loop_in_h => pad_loop_in_h,
    pad_rstn_ch0 => pad_rstn_ch0,
    pad_ddr_clk_w => pad_ddr_clk_w,
    pad_ddr_clkn_w => pad_ddr_clkn_w,
    pad_csn_ch0 => pad_csn_ch0,
    pad_addr_ch0 => pad_addr_ch0,
    pad_dq_ch0 => pad_dq_ch0,
    pad_dqs_ch0 => pad_dqs_ch0,
    pad_dqsn_ch0 => pad_dqsn_ch0,
    pad_dm_rdqs_ch0 => pad_dm_rdqs_ch0,
    pad_cke_ch0 => pad_cke_ch0,
    pad_odt_ch0 => pad_odt_ch0,
    pad_rasn_ch0 => pad_rasn_ch0,
    pad_casn_ch0 => pad_casn_ch0,
    pad_wen_ch0 => pad_wen_ch0,
    pad_ba_ch0 => pad_ba_ch0,
    pad_loop_out => pad_loop_out,
    pad_loop_out_h => pad_loop_out_h,
    areset_1 => areset_1,
    aclk_1 => aclk_1,
    awid_1 => awid_1,
    awaddr_1 => awaddr_1,
    awlen_1 => awlen_1,
    awsize_1 => awsize_1,
    awburst_1 => awburst_1,
    awlock_1 => awlock_1,
    awvalid_1 => awvalid_1,
    awready_1 => awready_1,
    awurgent_1 => awurgent_1,
    awpoison_1 => awpoison_1,
    wdata_1 => wdata_1,
    wstrb_1 => wstrb_1,
    wlast_1 => wlast_1,
    wvalid_1 => wvalid_1,
    wready_1 => wready_1,
    bid_1 => bid_1,
    bresp_1 => bresp_1,
    bvalid_1 => bvalid_1,
    bready_1 => bready_1,
    arid_1 => arid_1,
    araddr_1 => araddr_1,
    arlen_1 => arlen_1,
    arsize_1 => arsize_1,
    arburst_1 => arburst_1,
    arlock_1 => arlock_1,
    arvalid_1 => arvalid_1,
    arready_1 => arready_1,
    arpoison_1 => arpoison_1,
    rid_1 => rid_1,
    rdata_1 => rdata_1,
    rresp_1 => rresp_1,
    rlast_1 => rlast_1,
    rvalid_1 => rvalid_1,
    rready_1 => rready_1,
    arurgent_1 => arurgent_1,
    csysreq_1 => csysreq_1,
    csysack_1 => csysack_1,
    cactive_1 => cactive_1
  );
