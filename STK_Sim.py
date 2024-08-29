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
        res = self.root.ExecuteCommand("Parallel / AutomaticallyComputeInParallel On")
        res = self.root.ExecuteCommand(f"Parallel / Configuration ParallelType Local NumberOfLocalCores {os.cpu_count()}")

    def Target_Loader(self,Filename):
        self.targets = {}
        self.target_bins = []
        self.target_times = []
        for target in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            target.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for target_num in range(len(data)):
            self.targets[target_num] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f"Target_{target_num}")
            # IAgFacility target: Target Object
            self.targets[target_num].Position.AssignGeodetic(float(data['Lat'][target_num]), float(data['Lon'][target_num]), 0)  # Latitude, Longitude, Altitude
            # Set altitude to height of terrain
            self.targets[target_num].UseTerrain = True
            # Set altitude to a distance above the ground
            self.targets[target_num].HeightAboveGround = 0   # km
            self.target_bins.append(np.zeros([36,9]))
            self.target_times.append(self.root.CurrentScenario.StopTime)
            
    def Satellite_Loader(self,Filename,External_Pointing_File=False):
        self.satellites = {}
        self.radars = {}
        for satellite in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            satellite.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for satellite_num in range(len(data)):
            self.satellites[satellite_num] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f"Satellite_{satellite_num}")

            # IAgSatellite satellite: Satellite object
            keplerian = self.satellites[satellite_num].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
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
            self.satellites[satellite_num].Propagator.InitialState.Representation.Assign(keplerian)
            self.satellites[satellite_num].Propagator.Propagate()

            # IAgSatellite satellite: Satellite object
            # self.radars[satellite_num] = self.satellites[satellite_num].Children.New(AgESTKObjectType.eRadar, f'Radar{i+1}')
            # self.radars[satellite_num].CommonTasks.SetPatternSimpleConic(5, 0.1)
            # self.radars[satellite_num].CommonTasks.SetPatternSAR(0,90,0,0,data['Per'][i])
            # self.radars[satellite_num].SetPointingType(5)
            # for j in self.targets:
            #     self.radars[satellite_num].Pointing.Targets.Add(f'*/Target/{j}')

    def Update_Target_Bins(self,Interval):
        for bin in Interval.bins:
            if self.target_bins[Interval.target_number][bin//9,bin%9] == 0:
                self.target_times[Interval.target_number] = Interval.stop
            self.target_bins[Interval.target_number][bin//9,bin%9]+=1
        return 0

    def Compute_AzEl(self):
        self.Intervals = []
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            for tar_num,tar in enumerate(self.targets):
                for sat_num,sat in enumerate(self.satellites):
                    access = self.targets[tar].GetAccessToObject(self.satellites[sat])
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
    
    # def Interpolate_AzEl(self,interpolate_dt=2.5):
    #     interpolated_df = {'Time':[],'Satellite':[],'Target':[],'Azimuth':[],'Elevation':[],'Group':[]}
    #     n_sats = len(self.satellites)
    #     n_targets = len(self.targets)
    #     n_groups = len(np.unique(self.AzEl_data['Group'].values))
    #     with alive_bar(n_sats*n_targets*n_groups,force_tty=True,bar='classic',title='- Interp_AzEl   ',length=10) as bar:
    #         for sat_num in range(n_sats):
    #             sat_window = self.AzEl_data['Satellite'] == sat_num+1
    #             for tar_num in range(n_targets):
    #                 tar_window = self.AzEl_data['Target'] == tar_num+1
    #                 for group in range(n_groups):
    #                     group_window = self.AzEl_data['Group'].values==group+1
    #                     if len(self.AzEl_data[sat_window&tar_window&group_window]) > 0:
    #                         group_df = self.AzEl_data[sat_window&tar_window&group_window]
    #                         t = group_df['Time'].values
    #                         az = group_df['Azimuth'].values
    #                         el = group_df['Elevation'].values
                            
    #                         times = np.arange(t[0],t[-1],interpolate_dt)
    #                         if max(el)>=60 and len(t)>3:
    #                             az_t = interpolate.interpn(points=[t],values=np.array([np.unwrap(az,period=360)]).T,xi=times,method='pchip')[:,0]
    #                             el_t = interpolate.interp1d(x=t,y=[el],kind='cubic')(times)[0]
    #                         else:
    #                             ans = interpolate.interp1d(x=t,y=[np.unwrap(az,period=360),el],kind='quadratic')(times).T
    #                             az_t = ans[:,0];el_t = ans[:,1]
    #                         for idx,t in enumerate(times):
    #                             interpolated_df['Time'].append(t)
    #                             interpolated_df['Satellite'].append(sat_num+1)
    #                             interpolated_df['Target'].append(tar_num+1)
    #                             interpolated_df['Azimuth'].append(az_t[idx]%360)
    #                             interpolated_df['Elevation'].append(el_t[idx])
    #                             interpolated_df['Group'].append(group+1)
    #                     bar()
    #     self.AzEl_data = pd.concat([self.AzEl_data,pd.DataFrame(interpolated_df)],ignore_index=True).sort_values(by='Time')
    #     return 0