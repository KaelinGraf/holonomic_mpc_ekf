# Supervisor output (brief Section 7). Published by mpc_node every tick (20 Hz),
# computed in the same tick as the MPC solve that uses it.
# The values must match dock_control.supervisor.Stage.

uint8 IDLE=0
uint8 ACQUIRE=1
uint8 APPROACH=2
uint8 FINAL=3
uint8 DOCKED=4
uint8 HOLD=5
uint8 ABORT=6

uint8 stage
float64[3] y_ref
float64 v_max
float64 r_max
bool command_enabled
