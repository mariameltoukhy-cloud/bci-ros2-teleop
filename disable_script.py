from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient().require("sim")
h = sim.getObject("/PioneerP3DX/Script")
try:
    sim.setObjectInt32Param(h, sim.scriptintparam_enabled, 0)
    print("script disabled")
except Exception as e:
    print("disable failed, removing instead:", e)
    sim.removeObjects([h])
    print("script removed")
