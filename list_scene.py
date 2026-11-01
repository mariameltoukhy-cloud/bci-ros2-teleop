from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient().require("sim")
for h in sim.getObjectsInTree(sim.handle_scene):
    print(h, sim.getObjectAlias(h, 1))
