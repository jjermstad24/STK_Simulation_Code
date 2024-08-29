class Check_Lifetime:
    def __init__(self,stk_object):
        self.sim = stk_object
    def run(self,Cd=2.2,Cr=1.0,DragArea=13.65,SunArea=15.43,Mass=1000.0):
        decay_data = {"Satellite":[],"Orbit":[],"Lifetime":[]}
        for sat in self.sim.satellites:
            cmd = ("SetLifetime */Satellite/Satellite_"+ str(sat) +
                    " DragCoeff " + str(Cd) +
                    " ReflectCoeff " + str(Cr) + 
                    " DragArea " + str(DragArea) + 
                    " SunArea " + str(SunArea) + 
                    " Mass " + str(Mass)
                    )
            self.sim.root.ExecuteCommand(cmd)
            cmd  = "Lifetime */Satellite/Satellite_" + str(sat)
            res = self.sim.root.ExecuteCommand(cmd)
            line = res.Item(0).split()
            decay_data["Satellite"].append(sat)
            if line[2] == 'not':
                decay_data["Orbit"].append('>99999')
                decay_data["Lifetime"].append('-------')
            elif line[2] == 'before':
                decay_data["Orbit"].append('0')
                decay_data["Lifetime"].append('-------')
            else:
                orbits = float(line[12])
                time = float(line[16])
                time_unit = line[17][0:-1]
                decay_data["Orbit"].append(orbits)
                decay_data["Lifetime"].append(f"{time} {time_unit}")
        return decay_data