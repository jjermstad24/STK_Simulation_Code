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
            self.target_times.append(np.zeros([36,9]))
            
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
            self.target_times[idx] = self.root.CurrentScenario.StopTime*np.ones([36,9])

    def Update_Target_Bins(self,time,bin,target_number):
        if time < self.target_times[target_number][bin//9,bin%9]:
            self.target_times[target_number][bin//9,bin%9] = time
        self.target_bins[target_number][bin//9,bin%9]+=1
        return 0

    def Compute_AzEl(self,enable_print=True):
        self.root.ExecuteCommand("ClearAllAccess /")
        az_range = list(range(0,360,10))
        el_range = list(range(0,90,10))

        self.Reset_Target_Bins()
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10,disable=not(enable_print)) as bar:
            for sat in self.satellites:
                for tar_num,tar in enumerate(self.targets):
                    access = tar.GetAccessToObject(sat)
                    access.ComputeAccess()
                    Intervals = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,
                                                                                                          self.root.CurrentScenario.StopTime,
                                                                                                          self.dt,['Time','Azimuth','Elevation']).Intervals
                    for Int in Intervals:
                        times_az = Int.MultipleThresholdCrossings("Azimuth",az_range)
                        times_el = Int.MultipleThresholdCrossings("Elevation",el_range)
                        for i,t_az in enumerate(times_az[1:]):
                            for j,t_el in enumerate(times_el[1:]):
                                if len(t_az)>0 and len(t_el)>0:
                                    for t in t_el:
                                        if (any(t[0] <= value <= t[1] for value in t_az[0]) or 
                                            any(t_az[0][0] <= value <= t_az[0][1] for value in t)):
                                            self.Update_Target_Bins(t_az[0][0],i*9+j,tar_num)
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

    def Generate_Pre_Planning_Data(self):
        self.root.ExecuteCommand("ClearAllAccess /")
        az_range = list(range(0,360,10))
        el_range = list(range(0,90,10))

        self.Reset_Target_Bins()

        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_Access',length=10) as bar:
            for sat_num,sat in enumerate(self.satellites):
                for tar_num,tar in enumerate(self.targets):
                    tar.GetAccessToObject(sat).ComputeAccess()
                    sat.GetAccessToObject(tar).ComputeAccess()
                    bar()

        self.Pre_Planning_Hash_Map = {idx:{bin_num:[] for bin_num in range(324)} for idx in range(len(self.targets))}

        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Getting_AzEl',length=10) as bar:
            for sat_num,sat in enumerate(self.satellites):
                for tar_num,tar in enumerate(self.targets):
                    access = tar.GetAccessToObject(sat)
                    Intervals = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,
                                                                                                self.root.CurrentScenario.StopTime,
                                                                                                self.dt,['Time','Azimuth','Elevation']).Intervals
                    times = []
                    bins = []
                    for Int in Intervals:
                        times_az = Int.MultipleThresholdCrossings("Azimuth",az_range)
                        times_el = Int.MultipleThresholdCrossings("Elevation",el_range)
                        for i,t_az in enumerate(times_az[1:]):
                            for j,t_el in enumerate(times_el[1:]):
                                if len(t_az)>0 and len(t_el)>0:
                                    for t in t_el:
                                        if (any(t[0] <= value <= t[1] for value in t_az[0]) or 
                                            any(t_az[0][0] <= value <= t_az[0][1] for value in t)):
                                            self.Update_Target_Bins(t_az[0][0],i*9+j,tar_num)
                                            time_range = [float(i) for i in np.append(np.arange(t_az[0][0],t_az[0][1],self.dt),t_az[0][1])]
                                            times.extend(time_range)
                                            bins.extend(len(time_range)*[i*9+j])
                                            
                    access = sat.GetAccessToObject(tar)
                    res = access.DataProviders.GetItemByName('Sat Angles Data').ExecSingleElementsArray(times,['Cross Track','Along Track'])
                    crosstrack = res.GetArray(0)
                    alongtrack = res.GetArray(1)
                    for b,t,ct,at in zip(bins,times,crosstrack,alongtrack):
                        self.Pre_Planning_Hash_Map[tar_num][b].append([t,ct,at,sat_num])
                    
                    bar()
                    
        with alive_bar(len(self.targets)*324,force_tty=True,bar='classic',title='- Sorting_Data',length=10) as bar:
            for tar_num in range(len(self.targets)):
                for bin_num in range(324):
                    a = np.array(self.Pre_Planning_Hash_Map[tar_num][bin_num],dtype=float)
                    if len(a) > 0:
                        self.Pre_Planning_Hash_Map[tar_num][bin_num] = a[a[:,0].argsort()]
                    bar()
        
        return 0
        
    def Plan(self,slew_rate,cone_angle,time_threshold=6000):
        satellite_specific_plan = {key:{"Time":[],"Target":[],"Bin Number":[],"Cross Range":[],"Along Range":[]} for key in range(len(self.satellites))}
        bins = np.reshape([[[count,tar_num,bin_num] for bin_num,count in enumerate(tar_bin.ravel())] for tar_num,tar_bin in enumerate(self.target_bins)],[len(self.targets)*324,3]).astype(int)
        with alive_bar(324*len(self.targets),force_tty=True,bar='classic',title=f'- Planning',length=10) as bar:
            for count,tar_num,bin_num in bins[bins[:,0].argsort()]:
                if count > 0:
                    result = get_best_available_access(satellite_specific_plan,self.Pre_Planning_Hash_Map[tar_num][bin_num],slew_rate,cone_angle,time_threshold)
                    if type(result)!=bool:
                        sat_num = int(result[3])
                        satellite_specific_plan[sat_num]["Time"].append(result[0])
                        satellite_specific_plan[sat_num]["Target"].append(tar_num)
                        satellite_specific_plan[sat_num]["Bin Number"].append(bin_num)
                        satellite_specific_plan[sat_num]["Cross Range"].append(result[1])
                        satellite_specific_plan[sat_num]["Along Range"].append(result[2])
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

    def set_sim_time(self,days=1, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        self.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
        start_time = time_convert(self.root.CurrentScenario.StartTime)
        duration = datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks)
        stop_time=(start_time+duration).strftime("%d %b %Y %H:%M:%S.%f")
        self.root.CurrentScenario.StopTime=stop_time
        self.root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")