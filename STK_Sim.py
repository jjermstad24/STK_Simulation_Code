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
import scipy.interpolate as interpolate

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

    def Target_Loader(self,Filename):
        self.targets = {}
        for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            i.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for i in range(len(data)):
            self.targets[f'Target{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f'Target{i+1}')
            # IAgFacility target: Target Object
            self.targets[f'Target{i+1}'].Position.AssignGeodetic(float(data['Lat'][i]), float(data['Lon'][i]), 0)  # Latitude, Longitude, Altitude
            # Set altitude to height of terrain
            self.targets[f'Target{i+1}'].UseTerrain = True
            # Set altitude to a distance above the ground
            self.targets[f'Target{i+1}'].HeightAboveGround = 0   # km
            
    def Satellite_Loader(self,Filename,External_Pointing_File=False):
        self.satellites = {}
        self.sensors = {}
        for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            i.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for i in range(len(data)):
            self.satellites[f'Satellite{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f'Satellite{i+1}')

            # IAgSatellite satellite: Satellite object
            keplerian = self.satellites[f'Satellite{i+1}'].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
            keplerian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeAltitude
            keplerian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
            keplerian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN

            # Assign the perigee and apogee altitude values:
            keplerian.SizeShape.PerigeeAltitude = float(data['Per'][i])      # km
            keplerian.SizeShape.ApogeeAltitude = float(data['Apo'][i])        # km

            # Assign the other desired orbital parameters:
            keplerian.Orientation.Inclination = float(data['Inc'][i])               # deg
            keplerian.Orientation.ArgOfPerigee = float(data['AoP'][i])        # deg
            keplerian.Orientation.AscNode.Value = float(data['Asc'][i])            # deg
            keplerian.Location.Value = float(data['Loc'][i])                             # deg

            # Apply the changes made to the satellite's state and propagate:
            self.satellites[f'Satellite{i+1}'].Propagator.InitialState.Representation.Assign(keplerian)
            self.satellites[f'Satellite{i+1}'].Propagator.Propagate()

            # IAgSatellite satellite: Satellite object
            # self.sensors[f'Satellite{i+1}'] = self.satellites[f'Satellite{i+1}'].Children.New(AgESTKObjectType.eSensor, f'Sensor{i+1}')
            # self.sensors[f'Satellite{i+1}'].CommonTasks.SetPatternSimpleConic(5, 0.1)
            # # self.sensors[f'Satellite{i+1}'].CommonTasks.SetPatternSAR(0,90,0,0,data['Per'][i])
            # self.sensors[f'Satellite{i+1}'].SetPointingType(5)
            # for j in self.targets:
            #     self.sensors[f'Satellite{i+1}'].Pointing.Targets.Add(f'*/Target/{j}')

    def Compute_AzEl(self):
        df = {'Time':[],'Satellite':[],'Target':[],'Azimuth':[],'Elevation':[],'Group':[]}
        with alive_bar(len(self.targets),force_tty=True,bar='classic',title='- Computing_AzEl',length=10) as bar:
            for tar_num,tar in enumerate(self.targets):
                for sat_num,sat in enumerate(self.satellites):
                    access = self.targets[tar].GetAccessToObject(self.satellites[sat])
                    access.ComputeAccess()
                    data = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,['Time','Azimuth','Elevation']).DataSets.ToArray()
                    for i in range(len(data)):
                        for j in range(0,len(data[i]),3):
                            data[i].insert(j+3+j//3,j//3+1)
                    data = np.array(data).ravel()
                    time = data[0::4]
                    az = data[1::4]
                    el = data[2::4]
                    group = data[3::4]
                    for idx in range(len(time)):
                        if time[idx] != None:
                            df['Time'].append(time[idx])
                            df['Satellite'].append(sat_num+1)
                            df['Target'].append(tar_num+1)
                            df['Azimuth'].append(np.abs(az[idx]))
                            df['Elevation'].append(np.abs(el[idx]))
                            df['Group'].append(group[idx])
                bar()
        self.AzEl_data = pd.DataFrame(df).sort_values('Time')
        return 0
    
    def Interpolate_AzEl(self,interpolate_dt=2.5):
        interpolated_df = {'Time':[],'Satellite':[],'Target':[],'Azimuth':[],'Elevation':[],'Group':[]}
        n_sats = len(self.satellites)
        n_targets = len(self.targets)
        n_groups = len(np.unique(self.AzEl_data['Group'].values))
        with alive_bar(n_sats*n_targets*n_groups,force_tty=True,bar='classic',title='- Interp_AzEl   ',length=10) as bar:
            for sat_num in range(n_sats):
                sat_window = self.AzEl_data['Satellite'] == sat_num+1
                for tar_num in range(n_targets):
                    tar_window = self.AzEl_data['Target'] == tar_num+1
                    for group in range(n_groups):
                        group_window = self.AzEl_data['Group'].values==group+1
                        if len(self.AzEl_data[sat_window&tar_window&group_window]) > 0:
                            group_df = self.AzEl_data[sat_window&tar_window&group_window]
                            t = group_df['Time'].values
                            az = group_df['Azimuth'].values
                            el = group_df['Elevation'].values
                            
                            times = np.arange(t[0],t[-1],interpolate_dt)
                            if max(el)>=60 and len(t)>3:
                                az_t = interpolate.interpn(points=[t],values=np.array([np.unwrap(az,period=360)]).T,xi=times,method='pchip')[:,0]
                                el_t = interpolate.interp1d(x=t,y=[el],kind='cubic')(times)[0]
                            else:
                                ans = interpolate.interp1d(x=t,y=[np.unwrap(az,period=360),el],kind='quadratic')(times).T
                                az_t = ans[:,0];el_t = ans[:,1]
                            for idx,t in enumerate(times):
                                interpolated_df['Time'].append(t)
                                interpolated_df['Satellite'].append(sat_num+1)
                                interpolated_df['Target'].append(tar_num+1)
                                interpolated_df['Azimuth'].append(az_t[idx]%360)
                                interpolated_df['Elevation'].append(el_t[idx])
                                interpolated_df['Group'].append(group+1)
                        bar()
        self.AzEl_data = pd.concat([self.AzEl_data,pd.DataFrame(interpolated_df)],ignore_index=True).sort_values(by='Time')
        # self.AzEl_data['Satellite'] = pd.to_numeric(self.AzEl_data['Satellite'], downcast='integer')
        # self.AzEl_data['Target'] = pd.to_numeric(self.AzEl_data['Target'], downcast='integer')
        # self.AzEl_data['Group'] = pd.to_numeric(self.AzEl_data['Group'], downcast='integer')
        return 0
            
    def Sort_AzEl(self):
        self.Targets_Point_Bins = {}
        for tar in self.targets:
            self.Targets_Point_Bins[tar] = np.zeros([36,9])
        Bin = [];Point = [];Target_Percentage = []
        with alive_bar(len(self.AzEl_data),force_tty=True,bar='classic',title='- Sorting_AzEl  ',length=10) as bar:
            for idx in range(len(self.AzEl_data)):
                r,c = int(self.AzEl_data['Azimuth'].values[idx]//10),int(self.AzEl_data['Elevation'].values[idx]//10)
                Bin.append(r*9+c)
                tar = f"Target{self.AzEl_data['Target'].values[idx]}"
                Point.append(1/((self.Targets_Point_Bins[tar][r,c]+1)*(self.Targets_Point_Bins[tar][r,c]+1)))
                self.Targets_Point_Bins[tar][r,c] += Point[-1]
                Target_Percentage.append(100*np.count_nonzero(self.Targets_Point_Bins[tar])/324)
                bar()
        self.AzEl_data.loc[:,"Bin"] = Bin
        self.AzEl_data.loc[:,"Point"] = Point
        self.AzEl_data.loc[:,"Target Percentage"] = Target_Percentage
        return 0

    def Compute_Orientation(self):
        df = {'Time':[],'Satellite':[],'Yaw':[],'Pitch':[],'Roll':[],'Yaw Rate':[],'Pitch Rate':[],'Roll Rate':[]}
        for sat_num,sat in enumerate(self.satellites):
            data = np.array(self.satellites[sat].DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,['Time','YPR321 yaw','YPR321 pitch','YPR321 roll','YPR321 yaw rate','YPR321 pitch rate','YPR321 roll rate']).DataSets.ToArray()).ravel()
            time = data[0::7]
            yaw = data[1::7]
            pitch = data[2::7]
            roll = data[3::7]
            yaw_rate = data[4::7]
            pitch_rate = data[5::7]
            roll_rate = data[6::7]
            for idx in range(len(time)):
                if time[idx] != None:
                    df['Time'].append(time[idx])
                    df['Satellite'].append(sat_num+1)
                    df['Yaw'].append(yaw[idx])
                    df['Pitch'].append(pitch[idx])
                    df['Roll'].append(roll[idx])
                    df['Yaw Rate'].append(yaw_rate[idx])
                    df['Pitch Rate'].append(pitch_rate[idx])
                    df['Roll Rate'].append(roll_rate[idx])
        self.Orientation_data = pd.DataFrame(df).sort_values('Time')
        return 0

    def Compute_Lifetime(self,Cd=2.2,Cr=1.0,DragArea=13.65,SunArea=15.43,Mass=1000.0):
        labels=['SAT##','Orbits','Time']
        table = []
        for i in self.satellites:
            cmd = ("SetLifetime */Satellite/"+ i +
                    " DragCoeff " + str(Cd) +
                    " ReflectCoeff " + str(Cr) + 
                    " DragArea " + str(DragArea) + 
                    " SunArea " + str(SunArea) + 
                    " Mass " + str(Mass)
                    )
            self.root.ExecuteCommand(cmd)
            cmd  = "Lifetime */Satellite/" + i
            res = self.root.ExecuteCommand(cmd)
            line = res.Item(0).split()
            if line[2] == 'not':
                table.append([i[i.find('satellites'):],str('>99999'),'-------'])
            elif line[2] == 'before':
                table.append([i[i.find('satellites'):],0,0])
            else:
                orbits = float(line[12])
                years = float(line[16])
                time_unit = line[17][0:-1]
                table.append([i[i.find('satellites'):],orbits,str(years)+' '+str(time_unit)])
        self.Lifetimes = pd.DataFrame(table,columns=labels)
        self.Lifetimes.set_index('SAT##',inplace=True)