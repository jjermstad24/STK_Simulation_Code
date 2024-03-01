# STK library imports
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkengine import STKEngine
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.vgt import *
from Scripts import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# if using astrogator uncomment the below
# from agi.stk12.stkobjects.astrogator
# if using aviator uncomment the below
# from agi.stk12.stkobjects.aviator

class STK_Simulation:
    def __init__(self,GUI,Filename):
        STKDesktop.ReleaseAll()
        print("Loading STK...")
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
        print("Loading Scenario...")
        #Creating a new scenario
        self.scenario = self.root.NewScenario(Filename)

    def Target_Loader(self,Filename):
        print("Loading Targets",end='\r')
        self.targets = {}
        for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            i.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for i in range(len(data)):
            print(f'Loading Target{i+1}',end='\r')
            self.targets[f'Target{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f'Target{i+1}')
            # IAgFacility target: Target Object
            self.targets[f'Target{i+1}'].Position.AssignGeodetic(float(data['Lat'][i]), float(data['Lon'][i]), float(data['Alt'][i]))  # Latitude, Longitude, Altitude
            # Set altitude to height of terrain
            self.targets[f'Target{i+1}'].UseTerrain = True
            # Set altitude to a distance above the ground
            self.targets[f'Target{i+1}'].HeightAboveGround = 0   # km
        print("Loaded Targets")
            
    def Satellite_Loader(self,Filename,External_Pointing_File=False):
        print("Loading Satellites",end='\r')
        self.satellites = {}
        self.sensors = {}
        for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            i.Unload()

        data = pd.read_csv(Filename,delimiter=',')

        for i in range(len(data)):
            print(f'Loading Satellite{i+1}',end='\r')
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
            self.sensors[f'Satellite{i+1}'] = self.satellites[f'Satellite{i+1}'].Children.New(AgESTKObjectType.eSensor, f'Sensor{i+1}')
            self.sensors[f'Satellite{i+1}'].CommonTasks.SetPatternSimpleConic(5.0,0.1)
            if External_Pointing_File:
                self.sensors[f'Satellite{i+1}'].SetPointingExternalFile(External_Pointing_File)
            elif data['Tar'][i]:
                self.sensors[f'Satellite{i+1}'].SetPointingType(5)
                for j in self.targets:
                    self.sensors[f'Satellite{i+1}'].Pointing.Targets.Add(f'*/Target/{j}')
        print(f'Loaded Satellites')

    def Compute_AzEl(self,dt):
        print("Computing AzEl",end='\r')
        self.AzEl_data = {}
        self.Azimuth_vs_Elevation = {}
        for j in self.targets:
            self.Azimuth_vs_Elevation[j] = np.zeros([9,18])
            print(f'{j}...',end='\r')
            for i in self.satellites:
                access = self.targets[j].GetAccessToObject(self.sensors[i])
                access.ComputeAccess()
                DP = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,dt,['Time','Elevation','Azimuth'])
                data = np.array(DP.DataSets.ToArray())
                data_list=[]
                if len(data) > 0:
                    for k in range(np.shape(data)[1]//3):
                        data_list.append(np.array(data[0:,k*3:(k+1)*3]))
                    data_array = list(np.vstack(tuple(data_list)))
                    data_array = np.array([v for v in data_array if not(None in v)])
                    for k in range(len(data_array[:,0])):
                        data_array[k,0] = time_convert(data_array[k,0])
                        data_array[k,1] = float(data_array[k,1])
                        data_array[k,2] = float(data_array[k,2])
                    df = pd.DataFrame({'Time':data_array[:,0],'Elevation':data_array[:,1],'Azimuth':data_array[:,2]})
                    self.AzEl_data[f'{j}->{i}'] = df
                    y=np.linspace(0,180,19)
                    self.Azimuth_vs_Elevation[j]+=np.histogram2d(df['Azimuth'].astype(float),df['Elevation'].astype(float), bins=y, density=False)[0].tolist()[0:9]
            self.Azimuth_vs_Elevation[j] = pd.DataFrame(self.Azimuth_vs_Elevation[j])
            self.Azimuth_vs_Elevation[j].columns = [f"{i*10}-{(i+1)*10}" for i in range(0,18)]
            self.Azimuth_vs_Elevation[j].index = [f"{i*10}-{(i+1)*10}" for i in range(0,9)]
        print("Computed AzEl")
    
    def Compute_Lifetime(self,Cd=2.2,Cr=1.0,DragArea=13.65,SunArea=15.43,Mass=1000.0):
        print("Computing Lifetimes",end='\r')
        labels=['SAT##','Orbits','Time']
        table = []
        for i in self.satellites:
            print(f'{i}...',end='\r')
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
        print("Computed Lifetimes")
