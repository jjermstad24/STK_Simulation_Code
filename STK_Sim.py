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
        self.root.ExecuteCommand(f"Parallel / Configuration ParallelType Local NumberOfLocalCores {os.cpu_count()}")

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

    def Update_Target_Bins(self,time,bins,target_number):
        for bin in bins:
            if self.target_bins[target_number][bin//9,bin%9] == 0:
                self.target_times[target_number] = time[-1]
            self.target_bins[target_number][bin//9,bin%9]+=1
        return 0

    def Compute_AzEl(self):
        self.Reset_Target_Bins()
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            for sat in self.satellites:
                for tar_num,tar in enumerate(self.targets):
                    access = tar.GetAccessToObject(sat)
                    access.ComputeAccess()
                    DataSets = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,
                                                                                                        self.root.CurrentScenario.StopTime,
                                                                                                        self.dt,['Time','Azimuth','Elevation']).DataSets
                    for idx in range(0,DataSets.Count,3):
                        time = DataSets.Item(idx).GetValues()
                        az = DataSets.Item(idx+1).GetValues()
                        el = DataSets.Item(idx+2).GetValues()
                        if self.Interpolate:
                            time,az,el = Interpolate(time,az,el)
                        bins = np.unique([(a//10)*9+(e//10) for a,e in zip(az,el)]).astype(int)
                        self.Update_Target_Bins(time,bins,tar_num)
                    bar()
        return 0
    
    def Get_Satellite_DP(self,bus_name):
        dfs = []
        splits = bus_name.split("/")
        with alive_bar(len(self.satellites),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10) as bar:
            for sat in self.satellites:
                if len(splits) == 2:
                    bus = sat.DataProviders.GetItemByName(splits[0]).Group.GetItemByName(splits[1])
                elif len(splits) == 1:
                    bus = sat.DataProviders.GetItemByName(bus_name)
                df = bus.Exec(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt).DataSets.ToPandasDataFrame()
                for col in df.columns:
                    df[col] = df[col].astype(float,errors='ignore')
                dfs.append(df)
                bar()
        return dfs
    
    def Update_Mass_Properties(self,M=250,I=[[288,0,0],
                                             [0,88.88,0],
                                             [0,0,288]]):
        for sat in self.satellites:
            sat.MassProperties.Mass = M
            sat.MassProperties.Inertia.Ixx = I[0][0]
            sat.MassProperties.Inertia.Ixz = I[1][0]
            sat.MassProperties.Inertia.Iyy = I[1][1]
            sat.MassProperties.Inertia.Iyz = I[2][0]
            sat.MassProperties.Inertia.Izz = I[2][2]