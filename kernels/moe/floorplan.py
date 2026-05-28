from rapidstream import FloorplanConfig

config = FloorplanConfig(
    port_pre_assignments={".*": "SLOT_X0Y0:SLOT_X1Y0"},
    dse_range_min=0.55,
    dse_range_max=0.75,
)
config.save_to_file("floorplan_config.json")