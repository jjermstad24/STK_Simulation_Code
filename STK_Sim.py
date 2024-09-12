# STK library imports
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkengine import STKEngine
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.vgt import *
from Scripts import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alive_progress import alive_bar

# if using astrogator uncomment the below
# from agi.stk12.stkobjects.astrogator
# if using aviator uncomment the below
# from agi.stk12.stkobjects.aviator

class STK_Simulation:
    def __init__(self,GUI,Filename):
        STKDesktop.ReleaseAll()
        if GUI == True:
            # Start new instance of STK with GUI
            self.stk = STKDesktop.StartApplication(visible=True) #using optional visible argument
            # Get the IAgStkObjectRoot interface
            self.root = self.stk.Root
        else:
            # Start new instance of STK Engine with no GUI
            self.stk = STKEngine.StartApplication(noGraphics=True) # optionally, noGraphics = True
            # Get the IAgStkObjectRoot interface
            self.root = self.stk.NewObjectRoot()
        #Creating a new scenario
        self.scenario = self.root.NewScenario(Filename)
        self.dt = 60
        self.Interpolate = False
        self.root.ExecuteCommand("Parallel / AutomaticallyComputeInParallel On")
        self.root.ExecuteCommand(f"Parallel / Configuration ParallelType Local NumberOfLocalCores 16")

    def Target_Loader(self,Filename):
        self.targets = []
        self.target_bins = []
        self.target_times = []
        for target in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            target.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for target_num in range(len(data)):
            self.targets.append(self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f"Target_{target_num}"))
            # IAgFacility target: Target Object
            self.targets[-1].Position.AssignGeodetic(float(data['Lat'][target_num]), float(data['Lon'][target_num]), 0)  # Latitude, Longitude, Altitude
            # Set altitude to height of terrain
            self.targets[-1].UseTerrain = True
            # Set altitude to a distance above the ground
            self.targets[-1].HeightAboveGround = 0   # km
            self.target_bins.append(np.zeros([36,9]))
            self.target_times.append(self.root.CurrentScenario.StopTime)
            
    def Satellite_Loader(self,Filename,External_Pointing_File=False):
        self.satellites = []
        self.radars = []
        for satellite in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            satellite.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for satellite_num in range(len(data)):
            self.satellites.append(self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f"Satellite_{satellite_num}"))

            # IAgSatellite satellite: Satellite object
            keplerian = self.satellites[-1].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
            keplerian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeAltitude
            keplerian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
            keplerian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN

            # Assign the perigee and apogee altitude values:
            keplerian.SizeShape.PerigeeAltitude = float(data['Per'][satellite_num])      # km
            keplerian.SizeShape.ApogeeAltitude = float(data['Apo'][satellite_num])        # km

            # Assign the other desired orbital parameters:
            keplerian.Orientation.Inclination = float(data['Inc'][satellite_num])               # deg
            keplerian.Orientation.ArgOfPerigee = float(data['AoP'][satellite_num])        # deg
            keplerian.Orientation.AscNode.Value = float(data['Asc'][satellite_num])            # deg
            keplerian.Location.Value = float(data['Loc'][satellite_num])                             # deg

            # Apply the changes made to the satellite's state and propagate:
            self.satellites[-1].Propagator.InitialState.Representation.Assign(keplerian)
            self.satellites[-1].Propagator.Propagate()

            # IAgSatellite satellite: Satellite object
            # self.radars[satellite_num] = self.satellites[satellite_num].Children.New(AgESTKObjectType.eRadar, f'Radar{i+1}')
            # self.radars[satellite_num].CommonTasks.SetPatternSimpleConic(5, 0.1)
            # self.radars[satellite_num].CommonTasks.SetPatternSAR(0,90,0,0,data['Per'][i])
            # self.radars[satellite_num].SetPointingType(5)
            # for j in self.targets:
            #     self.radars[satellite_num].Pointing.Targets.Add(f'*/Target/{j}')

    def Reset_Target_Bins(self):
        for idx in range(len(self.targets)):
            self.target_bins[idx] = np.zeros([36,9])
            self.target_times[idx] = self.root.CurrentScenario.StopTime

    def Update_Target_Bins(self,Interval):
        for bin in Interval.bins:
            if self.target_bins[Interval.target_number][bin//9,bin%9] == 0:
                self.target_times[Interval.target_number] = Interval.stop
            self.target_bins[Interval.target_number][bin//9,bin%9]+=1
        return 0

    def Compute_AzEl(self):
        self.Reset_Target_Bins()
        self.Intervals = []
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            for tar_num,tar in enumerate(self.targets):
                for sat_num,sat in enumerate(self.satellites):
                    access = tar.GetAccessToObject(sat)
                    access.ComputeAccess()
                    data_set = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,['Time','Azimuth','Elevation']).DataSets
                    data = data_set.ToNumpyArray()
                    if len(data) > 0:
                        groups = data_set.Count//3
                        for idx in range(groups):
                            access_point = data[idx::groups]
                            access_point = np.abs(access_point[access_point[:,0]!=None]).astype(float)
                            I = Interval(access_point,tar_num,sat_num,self.Interpolate)
                            self.Intervals.append(I)
                            self.Update_Target_Bins(I)
                    bar()
        return 0
    
    def Compute_AzEl_No_Interpolation(self):
        self.Reset_Target_Bins()
        self.Intervals = []
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            for tar_num,tar in enumerate(self.targets):
                for sat_num,sat in enumerate(self.satellites):
                    access = tar.GetAccessToObject(sat)
                    access.ComputeAccess()
                    data_set = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,['Time','Azimuth','Elevation']).DataSets
                    data = data_set.ToNumpyArray()
                    data = np.array([row for row in data if None not in row])
                    if len(data) > 0:
                        self.bins = np.unique([int((float(az)//10)*9) + int(float(el)//10) for az,el in zip(data[:,1],data[:,2])])
                        self.target_bins[tar_num][self.bins//9,self.bins%9]+=1
                    bar()
        return 0

    
    
    def Create_TarCon(self):
        self.constellation = self.root.CurrentScenario.Children.New(AgESTKObjectType.eConstellation, "TarCon")
        for tar_num, tar in enumerate(self.targets):
            self.constellation.Objects.AddObject(tar)
        return 0
    
    def Constellation_and_Chain_Loader(self,pop,write=True):
        for chain in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eChain):
            chain.Unload()
        self.chain = self.root.CurrentScenario.Children.New(AgESTKObjectType.eChain,'Targets_to_Constellations')
        self.chain.AutoRecompute = False

        self.constellations = []
        for constellation in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eConstellation):
            constellation.Unload()

        for satellite in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            satellite.Unload()

        for satcon_num, sat_con_data in enumerate(pop):
            Individual = sat_con_data
            self.constellation = self.root.CurrentScenario.Children.New(AgESTKObjectType.eConstellation, f"SatCon_{satcon_num}")
            self.constellations.append(self.constellation)
            self.root.ExecuteCommand(f"Chains */Chain/Targets_to_Constellations Connections Add from Constellation/TarCon to Constellation/SatCon_{satcon_num}")

            Alt = Individual[0]
            Inc = Individual[1]
            Aop = Individual[2]
            num_sats = int(Individual[3])
            num_planes = int(Individual[4])

            if num_planes <= num_sats:
                if write:
                    file = open(f"Input_Files/SatCon_{satcon_num}.txt","w")
                    file.write("Per,Apo,Inc,AoP,Asc,Loc,Tar\n")
                    sats = num_sats*[1]
                    planes = np.array_split(sats,num_planes)
                    Asc = 0
                    for plane in planes:
                        Loc = 0
                        for sat in plane:
                            file.write(f"{Alt},{Alt},{Inc},{Aop},{round(Asc,4)},{round(Loc,4)},{1}\n")
                            if len(plane)>1: Loc += 360/(len(plane)-1)
                        if len(planes)>1:Asc += 180/(len(planes)-1)
                    file.close()
                constellation_filename = f"Input_Files/SatCon_{satcon_num}.txt"

            self.satellites = []
            self.radars = []

            data = pd.read_csv(constellation_filename,delimiter=',')

            for satellite_num in range(len(data)):
                sat_num = satellite_num
                sat_num += 100*satcon_num

                sat = self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f"Satellite_{sat_num}")
                self.satellites.append(sat)

                self.constellation.Objects.AddObject(sat)

                # IAgSatellite satellite: Satellite object
                keplerian = self.satellites[-1].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
                keplerian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeAltitude
                keplerian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
                keplerian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN

                # Assign the perigee and apogee altitude values:
                keplerian.SizeShape.PerigeeAltitude = float(data['Per'][satellite_num])      # km
                keplerian.SizeShape.ApogeeAltitude = float(data['Apo'][satellite_num])        # km

                # Assign the other desired orbital parameters:
                keplerian.Orientation.Inclination = float(data['Inc'][satellite_num])               # deg
                keplerian.Orientation.ArgOfPerigee = float(data['AoP'][satellite_num])        # deg
                keplerian.Orientation.AscNode.Value = float(data['Asc'][satellite_num])            # deg
                keplerian.Location.Value = float(data['Loc'][satellite_num])                             # deg

                # Apply the changes made to the satellite's state and propagate:
                self.satellites[-1].Propagator.InitialState.Representation.Assign(keplerian)
                self.satellites[-1].Propagator.Propagate()

                # IAgSatellite satellite: Satellite object
                # self.radars[satellite_num] = self.satellites[satellite_num].Children.New(AgESTKObjectType.eRadar, f'Radar{i+1}')
                # self.radars[satellite_num].CommonTasks.SetPatternSimpleConic(5, 0.1)
                # self.radars[satellite_num].CommonTasks.SetPatternSAR(0,90,0,0,data['Per'][i])
                # self.radars[satellite_num].SetPointingType(5)
                # for j in self.targets:
                #     self.radars[satellite_num].Pointing.Targets.Add(f'*/Target/{j}')


        self.root.ExecuteCommand(f"Chains */Chain/Targets_to_Constellations Connections SetStartInst Constellation/TarCon")
        self.root.ExecuteCommand(f"Chains */Chain/Targets_to_Constellations Connections SetEndInst Constellation/SatCon_{satcon_num}")
                

    def Compute_Chain_AzEl(self):
        self.Reset_Target_Bins()
        self.chain.ComputeAccess()

        with alive_bar(len(self.constellations),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            data_set = self.chain.DataProviders.GetItemByName("Access AER Data").ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,['Time','Azimuth','Elevation', 'Strand Name']).DataSets
            data = data_set.ToNumpyArray()
            for satcon_num,sat in enumerate(self.constellations):
                self.Reset_Target_Bins()
                
                for tar_num,tar in enumerate(self.targets):
                    tar_data = data[
                    (np.char.find(data[:, 3].astype(str), f'Target_{tar_num} ') != -1) & (np.char.find(data[:, 3].astype(str), f'Satellite_{satcon_num}') != -1) ]
                    if len(tar_data) > 0:
                        self.bins = np.unique([int((az//10)*9) + int(el//10) for az,el in zip(tar_data[:,1].astype(float),tar_data[:,2].astype(float))])
                        self.target_bins[tar_num][self.bins//9,self.bins%9]+=1
                    self.percentages = [100*np.count_nonzero(self.target_bins[idx])/324 for idx in range(len(self.targets))]
                    self.constellations[satcon_num]['Image']
                bar()
        

    
    def Get_Satellite_DP(self,bus_name):
        dfs = []
        with alive_bar(len(self.satellites),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10) as bar:
            for sat in self.satellites:
                bus = sat.DataProviders.GetItemByName(bus_name)
                df = bus.Exec(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt).DataSets.ToPandasDataFrame()
                for col in df.columns:
                    df[col] = df[col].astype(float,errors='ignore')
                dfs.append(df)
                bar()
        return dfs