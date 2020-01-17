run_type particle               # Monte Carlo run
output_prefix out/urban_plume # prefix of output files
n_repeat 1
n_part 5000
restart no                      # whether to restart from saved state (yes/no)
 
t_max 172800
del_t 60                        # timestep (s)
t_output 3600 
t_progress 14400 
 
n_bin 220                       # number of bins
d_min 1e-10                     # minimum diameter (m)
d_max 1e-4                      # maximum diameter (m)
 
weight none                     # unweighted particles
 
gas_data gas_data.dat           # file containing gas data
gas_init gas_init.dat           # initial gas concentrations
 
aerosol_data aero_data.dat      # file containing aerosol data
aerosol_init aero_init_dist.dat # aerosol initial condition file
 
temp_profile temp.dat   # temperature profile file
height_profile height.dat       # height profile file
gas_emissions gas_emit.dat      # gas emissions file
gas_background gas_back.dat     # background gas concentrations file
aero_emissions aero_emit.dat    # aerosol emissions file
aero_background aero_back.dat   # aerosol background file
 
rel_humidity 0.2865
pressure 1e5                    # initial pressure (Pa)
latitude 38.4579
longitude 0                     # longitude (degrees, -180 to 180)
altitude 0                      # altitude (m)
start_time 21600
start_day 353
 
do_coagulation yes 	 # whether to do coagulation (yes/no) 
coag_kernel brown 
do_condensation no              # whether to do condensation (yes/no)
do_mosaic yes                    # whether to do MOSAIC (yes/no)
do_optical yes                  # whether to compute optical props (yes/no)
do_nucleation no                # whether to do nucleation (yes/no)
 
rand_init 0                      # random initialization (0 to use time)
allow_doubling yes               # whether to allow doubling (yes/no)
allow_halving yes                # whether to allow halving (yes/no)
record_removals yes              # whether to record particle removals (yes/no)
do_parallel no
run_type particle               # Monte Carlo run
output_prefix out/urban_plume # prefix of output files
n_repeat 1
n_part 5000
restart no                      # whether to restart from saved state (yes/no)
 
t_max 172800
del_t 60                        # timestep (s)
t_output 600
t_progress 600                  # progress printing interval (0 disables) (s)
 
n_bin 220                       # number of bins
d_min 1e-10                     # minimum diameter (m)
d_max 1e-4                      # maximum diameter (m)
 
weight none                     # unweighted particles
 
gas_data gas_data.dat           # file containing gas data
gas_init gas_init.dat           # initial gas concentrations
 
aerosol_data aero_data.dat      # file containing aerosol data
aerosol_init aero_init_dist.dat # aerosol initial condition file
 
temp_profile temp.dat   # temperature profile file
height_profile height.dat       # height profile file
gas_emissions gas_emit.dat      # gas emissions file
gas_background gas_back.dat     # background gas concentrations file
aero_emissions aero_emit.dat    # aerosol emissions file
aero_background aero_back.dat   # aerosol background file
 
rel_humidity 0.2865
pressure 1e5                    # initial pressure (Pa)
latitude 38.4579
longitude 0                     # longitude (degrees, -180 to 180)
altitude 0                      # altitude (m)
start_time 21600
start_day 353
 
do_coagulation yes 	 # whether to do coagulation (yes/no) 
coag_kernel brown 
do_condensation no              # whether to do condensation (yes/no)
do_mosaic yes                    # whether to do MOSAIC (yes/no)
do_optical yes                  # whether to compute optical props (yes/no)
do_nucleation no                # whether to do nucleation (yes/no)
 
rand_init 0                      # random initialization (0 to use time)
allow_doubling yes               # whether to allow doubling (yes/no)
allow_halving yes                # whether to allow halving (yes/no)
record_removals yes              # whether to record particle removals (yes/no)
do_parallel no
