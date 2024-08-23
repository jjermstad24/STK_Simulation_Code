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

    def Target_Loader(self,Filename):
        self.targets = {}
        for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            i.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for i in range(len(data)):
            self.targets[f'Target{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f'Target{i+1}')
            # IAgFacility target: Target Object
            self.targets[f'Target{i+1}'].Position.AssignGeodetic(float(data['Lat'][i]), float(data['Lon'][i]), float(data['Alt'][i]))  # Latitude, Longitude, Altitude
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

    def Compute_AzEl(self,dt):
        self.AzEl_data = {}
        with alive_bar(len(self.targets),force_tty=True,bar='classic',title='Computing_AzEl') as bar:
            for j in self.targets:
                for i in self.satellites:
                    access = self.targets[j].GetAccessToObject(self.satellites[i])
                    access.ComputeAccess()
                    DP = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,dt,['Time','Elevation','Azimuth'])
                    data = np.array(DP.DataSets.ToArray())
                    data_list=[]
                    if len(data) > 0:
                        for k in range(np.shape(data)[1]//3):
                            data_list.append(np.array(data[0:,k*3:(k+1)*3]))
                        data_array = list(np.vstack(tuple(data_list)))
                        data_array = np.array([v for v in data_array if not(None in v)])
                        df = pd.DataFrame({'Time':data_array[:,0],'Azimuth':data_array[:,1],'Elevation':data_array[:,2]})
                        self.AzEl_data[f'{j}->{i}'] = df
                    else:
                        self.AzEl_data[f'{j}->{i}'] = 0
                bar()

    def Sort_AzEl_hist_per_target(self):
        self.AzEl_hist_per_target = {}
        x=np.arange(0,91,10)
        y=np.arange(0,361,10)
        with alive_bar(len(self.targets),force_tty=True,bar='classic',title='Sorting_AzEl_hist_per_target') as bar:
            for j in self.targets:
                self.AzEl_hist_per_target[j] = np.zeros([36,9])
                for i in self.satellites:
                    df = self.AzEl_data[f'{j}->{i}']
                    if type(df) != int:
                        self.AzEl_hist_per_target[j]+=np.histogram2d(np.abs(df['Azimuth'].astype(float)),np.abs(df['Elevation'].astype(float)), bins=[y,x], density=False)[0]
                self.AzEl_hist_per_target[j] = pd.DataFrame(self.AzEl_hist_per_target[j])
                self.AzEl_hist_per_target[j].columns = [f"{i*10}-{(i+1)*10}" for i in range(0,9)]
                self.AzEl_hist_per_target[j].index = [f"{i*10}-{(i+1)*10}" for i in range(0,36)]
                bar()

    def Sort_AzEl_4d_representation(self):
        self.AzEl_4d_representation = {}
        x=np.arange(0,91,10)
        y=np.arange(0,361,10)
        with alive_bar(len(self.satellites),force_tty=True,bar='classic',title='Sorting_AzEl_4d_representation') as bar:
            df = {'Time':[],'Satellite':[],'Target':[],'Bin':[]}
            for sat in self.satellites:
                for tar in self.targets:
                    target_df = self.AzEl_data[f'{tar}->{sat}']
                    if type(target_df)!=int:
                        for idx in range(len(target_df)):
                            df['Time'].append(target_df['Time'].values[idx])
                            df['Satellite'].append(int(sat.strip("Satellite")))
                            df['Target'].append(int(tar.strip("Target")))
                            Az = np.abs(target_df['Azimuth'].values[idx])
                            El = np.abs(target_df['Elevation'].values[idx])
                            r,c = np.where(np.histogram2d([Az],[El], bins=[y,x], density=False)[0]==1)
                            r = r[0];c = c[0]
                            df['Bin'].append(r*9+c)
                bar()
        df = pd.DataFrame(df)
        df = df.sort_values('Time')
        self.AzEl_4d_representation = df



    def Sort_AzEl_time_per_target(self):
        data = self.AzEl_data
        self.AzEl_time_per_target = {}
        with alive_bar(len(self.targets),force_tty=True,bar='classic',title='Sorting_AzEl_time_per_target') as bar:
            for t in self.targets:
                dfs = []
                for s in self.satellites:
                    if type(data[f'{t}->{s}']) != int:
                        dfs.append(data[f'{t}->{s}'])
                if len(dfs) > 0:
                    AzEl_time_per_target = pd.concat(dfs).sort_values(by="Time",ignore_index=True)
                    img_percent = []
                    Target_Angles = np.zeros([36,9])
                    for idx,time in enumerate(AzEl_time_per_target['Time']):
                        Az = AzEl_time_per_target['Azimuth'][idx]
                        El = AzEl_time_per_target['Elevation'][idx]
                        Target_Angles[int(Az//10),int(El//10)] += 1
                        num_total_angles = len(np.where(Target_Angles>0)[0])
                        img_percent.append((100*num_total_angles/324))
                    AzEl_time_per_target['Percent Imaged'] = img_percent
                    self.AzEl_time_per_target[t] = AzEl_time_per_target
                else:
                    self.AzEl_time_per_target[t] = 0
                bar()
    
    def Compute_YPR_rates(self,dt):
        self.YPR_rates_data = {}
        for j in self.sensors:
            DP = self.sensors[j].DataProviders.GetItemByName('Body Axes Orientation').Group.Item(4).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,dt,['Time','YPR321 yaw rate','YPR321 pitch rate','YPR321 roll rate'])
            data = np.array(DP.DataSets.ToArray())
            data_list=[]
            if len(data) > 0:
                for k in range(np.shape(data)[1]//4):
                    data_list.append(np.array(data[0:,k*4:(k+1)*4]))
                data_array = list(np.vstack(tuple(data_list)))
                data_array = np.array([v for v in data_array if not(None in v)])
                df = pd.DataFrame({'Time':data_array[:,0],'yaw rate':data_array[:,1],'pitch rate':data_array[:,2],'roll rate':data_array[:,3]})
                df['yaw rate'] = df['yaw rate'].astype(float)
                df['pitch rate'] = df['pitch rate'].astype(float)
                df['roll rate'] = df['roll rate'].astype(float)
                self.YPR_rates_data[f'{j}'] = df

    def Compute_YPR(self,dt):
        self.YPR_data = {}
        for j in self.sensors:
            DP = self.sensors[j].DataProviders.GetItemByName('Body Axes Orientation').Group.Item(4).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,dt,['Time','YPR321 yaw','YPR321 pitch','YPR321 roll'])
            data = np.array(DP.DataSets.ToArray())
            data_list=[]
            if len(data) > 0:
                for k in range(np.shape(data)[1]//4):
                    data_list.append(np.array(data[0:,k*4:(k+1)*4]))
                data_array = list(np.vstack(tuple(data_list)))
                data_array = np.array([v for v in data_array if not(None in v)])
                df = pd.DataFrame({'Time':data_array[:,0],'yaw':data_array[:,1],'pitch':data_array[:,2],'roll':data_array[:,3]})
                df['yaw'] = df['yaw'].astype(float)
                df['pitch'] = df['pitch'].astype(float)
                df['roll'] = df['roll'].astype(float)
                self.YPR_data[f'{j}'] = df

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