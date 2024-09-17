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

    def Update_Target_Bins(self,time,bin,target_number):
        if self.target_bins[target_number][bin//9,bin%9] == 0:
            self.target_times[target_number] = time
        self.target_bins[target_number][bin//9,bin%9]+=1
        return 0

    def Compute_AzEl(self,enable_print=True):
        self.Reset_Target_Bins()
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10,disable=not(enable_print)) as bar:
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
                        el = np.abs(DataSets.Item(idx+2).GetValues())
                        if self.Interpolate:
                            time,az,el = Interpolate(time,az,el)
                        bins = np.array([(a//10)*9+(e//10) for a,e in zip(az,el)]).astype(int)
                        for j in range(len(time)):
                            self.Update_Target_Bins(time[j],bins[j],tar_num)
                    bar()
        return 0
    
    def Get_Satellite_DP(self,bus_name,enable_print=True):
        dfs = []
        splits = bus_name.split("/")
        with alive_bar(len(self.satellites),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10,disable=not(enable_print)) as bar:
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
    
    def Get_Access_DP(self,obs1,obs2,bus_name,Total_Elements=False,enable_print=True):
        dfs = []
        splits = bus_name.split("/")
        with alive_bar(len(self.satellites)*len(self.targets),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10,disable=not(enable_print)) as bar:
            for ob1 in obs1:
                dfs.append([])
                for ob2 in obs2:
                    access = ob1.GetAccessToObject(ob2)
                    access.ComputeAccess()
                    if len(splits) == 2:
                        bus = access.DataProviders.GetItemByName(splits[0]).Group.GetItemByName(splits[1])
                    if len(splits) == 1:
                        bus = access.DataProviders.GetItemByName(splits[0])
                    df = {}
                    if not(Total_Elements):
                        DataSets = bus.Exec(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt).DataSets
                        Elements = DataSets.ElementNames
                    else:
                        DataSets = bus.ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,Total_Elements).DataSets
                        Elements = Total_Elements
                    if len(Elements) > 0:
                        for e in Elements:
                            df[e] = []
                        for idx in range(0,DataSets.Count,len(Elements)):
                            for e_idx,e in enumerate(Elements):
                                df[e].extend(DataSets.Item(idx+e_idx).GetValues())
                        dfs[-1].append(df)
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

    def Generate_Holding_Data(self):
        AER_Data = self.Get_Access_DP(self.targets,
                                      self.satellites,
                                      "AER Data/Default",
                                      ['Time','Azimuth','Elevation'])
        Sat_Angles_Data = self.Get_Access_DP(self.satellites,
                                             self.targets,
                                             "Sat Angles Data",
                                             ["Cross Track","Along Track"])
        
        self.Holding_Data = {key:[] for key in AER_Data[0][0]}
        self.Holding_Data['Satellite'] = []
        self.Holding_Data['Target'] = []
        self.Holding_Data['Cross Track'] = []
        self.Holding_Data['Along Track'] = []

        with alive_bar(len(self.targets),force_tty=True,bar='classic',title=f'- Compiling_Data',length=10) as bar:
            for tar_num,tar in enumerate(self.targets):
                data_holder = AER_Data[tar_num].copy()
                for sat_num,df in enumerate(data_holder):
                    df['Elevation'] = np.abs(df['Elevation'])
                    df['Target'] = tar_num*np.ones_like(df[list(df.keys())[0]]).astype(int)
                    df['Satellite'] = sat_num*np.ones_like(df[list(df.keys())[0]]).astype(int)
                    df['Cross Track'] = Sat_Angles_Data[sat_num][tar_num]['Cross Track']
                    df['Along Track'] = Sat_Angles_Data[sat_num][tar_num]['Along Track']
                    for key in self.Holding_Data:
                        self.Holding_Data[key].extend(df[key])
                bar()

        bins = []
        for idx in range(len(self.Holding_Data['Time'])):
            az = self.Holding_Data['Azimuth'][idx]
            el = self.Holding_Data['Elevation'][idx]
            bin = int(az//10*9+el//10)
            self.Update_Target_Bins(self.Holding_Data['Time'][idx],bin,self.Holding_Data['Target'][idx])
            bins.append(bin)

        self.Holding_Data["Bin Number"] = bins
        self.Holding_Data = pd.DataFrame(self.Holding_Data).sort_values(by="Time")

        self.hash_map = {}
        with alive_bar(324*len(self.targets),force_tty=True,bar='classic',title=f'- Creating_Hash_Map',length=10) as bar:
            for tar_num in range(len(self.targets)):
                self.hash_map[tar_num] = {}
                tar_window = self.Holding_Data['Target'].values==tar_num
                for bin_num in range(324):
                    bin_window = self.Holding_Data['Bin Number'].values==bin_num
                    self.hash_map[tar_num][bin_num] = np.array([self.Holding_Data['Time'].values[bin_window&tar_window],
                                                                self.Holding_Data['Satellite'].values[bin_window&tar_window].astype(int),
                                                                self.Holding_Data['Cross Track'].values[bin_window&tar_window],
                                                                self.Holding_Data['Along Track'].values[bin_window&tar_window]]).T
                    bar()
        return 0
        
    def Plan(self,slew_rate,cone_angle):
        satellite_specific_plan = {key:{"Time":[],"Target":[],"Bin Number":[],"Cross Range":[],"Along Range":[]} for key in range(len(self.satellites))}
        bins = np.reshape([[[count,tar_num,bin_num] for bin_num,count in enumerate(tar_bin.ravel())] for tar_num,tar_bin in enumerate(self.target_bins)],[len(self.targets)*324,3]).astype(int)
        with alive_bar(324*len(self.targets),force_tty=True,bar='classic',title=f'- Planning',length=10) as bar:
            for count,tar_num,bin_num in bins[bins[:,0].argsort()]:
                if count > 0:
                    result = get_best_available_access(satellite_specific_plan,self.hash_map[tar_num][bin_num],slew_rate,cone_angle)
                    if type(result)!=bool:
                        satellite_specific_plan[result[1]]["Time"].append(result[0])
                        satellite_specific_plan[result[1]]["Target"].append(tar_num)
                        satellite_specific_plan[result[1]]["Bin Number"].append(bin_num)
                        satellite_specific_plan[result[1]]["Cross Range"].append(result[2])
                        satellite_specific_plan[result[1]]["Along Range"].append(result[3])
                bar()
                
        Times = []
        Sats = []
        Cross_Range = []
        Along_Range = []
        Targets = []
        Bins = []
        for sat_num in satellite_specific_plan:
            Times.extend(satellite_specific_plan[sat_num]["Time"])
            Sats.extend(len(satellite_specific_plan[sat_num]["Time"])*[sat_num])
            Cross_Range.extend(satellite_specific_plan[sat_num]["Cross Range"])
            Along_Range.extend(satellite_specific_plan[sat_num]["Along Range"])
            Targets.extend(satellite_specific_plan[sat_num]["Target"])
            Bins.extend(satellite_specific_plan[sat_num]["Bin Number"])

        self.Planned_Data = pd.DataFrame({"Time":Times,
                                          "Satellite":Sats,
                                          "Target":Targets,
                                          "Cross Track":Cross_Range,
                                          "Along Track":Along_Range,
                                          "Bin Number":Bins})
        return 0