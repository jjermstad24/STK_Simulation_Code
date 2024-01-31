# STK library imports
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkengine import STKEngine
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.vgt import *
from Scripts import *
# if using astrogator uncomment the below
# from agi.stk12.stkobjects.astrogator
# if using aviator uncomment the below
# from agi.stk12.stkobjects.aviator

class STK_Simulation:
    def __init__(self,GUI,Filename):
        STKDesktop.ReleaseAll()
        print("Loading STK...")
        Print_Spacing()
        if GUI == True:
            # Start new instance of STK with GUI
            self.stk = STKDesktop.StartApplication(visible=True) #using optional visible argument
            # Get the IAgStkObjectRoot interface
            self.root = self.stk.Root
        else:
            # Start new instance of STK Engine with no GUI
            self.stk = STKEngine.StartApplication(noGraphics=False) # optionally, noGraphics = True
            # Get the IAgStkObjectRoot interface
            self.root = self.stk.NewObjectRoot()
        
        print("Loaded STK")
        Print_Spacing()
                
        print("Loading Scenario...")
        Print_Spacing()

        #Creating a new scenario
        self.scenario = self.root.NewScenario(Filename)

        print("Loaded Scenario")
        Print_Spacing()

    def Target_Loader(self,Target_Load_Option=0):
        print("Loading Targets...")
        Print_Spacing()
        if Target_Load_Option == 0:
            self.targets = {}
            for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
                i.Unload()
            for i in range(15):
                lon = float(decimal.Decimal(random.randrange(-900000, 900000))/10000)
                lat = float(decimal.Decimal(random.randrange(-900000, 900000))/10000)
                self.targets[f'Target{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f'Target{i+1}')
                # IAgFacility target: Target Object
                self.targets[f'Target{i+1}'].Position.AssignGeodetic(lon, lat, 0)  # Latitude, Longitude, Altitude
                # Set altitude to height of terrain
                self.targets[f'Target{i+1}'].UseTerrain = True
                # Set altitude to a distance above the ground
                self.targets[f'Target{i+1}'].HeightAboveGround = 0   # km
        print("Loaded Targets")
        Print_Spacing()
            
    def Satellite_Loader(self,Satellite_Load_Option=0,Num_Satellites=12,per_range=(100,1000),ago_range=(100,1000),
                         inc_range=(45,90),aop_range=(45,90),asc_range=(45,90),loc_range=(45,90)):
        print("Loading Satellites...")
        Print_Spacing()
        if Satellite_Load_Option == 0:
            self.satellites = {}
            for i in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
                i.Unload()

            for i in range(Num_Satellites):
                perigee_altitude = Random_Decimal(per_range)
                apogee_altitude = Random_Decimal(ago_range)
                
                inclination = Random_Decimal(inc_range)
                argument_of_perigee = Random_Decimal(aop_range)
                ascending_node = Random_Decimal(asc_range)
                location = Random_Decimal(loc_range)

                self.satellites[f'Satellite{i+1}'] = self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f'Satellite{i+1}')

                # IAgSatellite satellite: Satellite object
                keplerian = self.satellites[f'Satellite{i+1}'].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
                keplerian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeAltitude
                keplerian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
                keplerian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN

                # Assign the perigee and apogee altitude values:
                keplerian.SizeShape.PerigeeAltitude = perigee_altitude      # km
                keplerian.SizeShape.ApogeeAltitude = apogee_altitude        # km

                # Assign the other desired orbital parameters:
                keplerian.Orientation.Inclination = inclination                 # deg
                keplerian.Orientation.ArgOfPerigee = argument_of_perigee        # deg
                keplerian.Orientation.AscNode.Value = ascending_node            # deg
                keplerian.Location.Value = location                             # deg

                # Apply the changes made to the satellite's state and propagate:
                self.satellites[f'Satellite{i+1}'].Propagator.InitialState.Representation.Assign(keplerian)
                self.satellites[f'Satellite{i+1}'].Propagator.Propagate()

                # IAgSatellite satellite: Satellite object
                sensor = self.satellites[f'Satellite{i+1}'].Children.New(AgESTKObjectType.eSensor, f'Sensor{i+1}')
                sensor.CommonTasks.SetPatternSimpleConic(5.0,0.1)
                attitudePointing = self.satellites[f'Satellite{i+1}'].Attitude.Pointing
                attitudePointing.UseTargetPointing = True
        print("Loaded Satellites")
        Print_Spacing()

        def plot(self):
            fig = make_subplots(rows=5, cols=3,shared_xaxes=True,shared_yaxes=True)
            for j in self.targets:
                targetCoverage = self.targets[j].ObjectCoverage
                targetCoverage.Assets.RemoveAll()
                for i in self.satellites:
                    targetCoverage.Assets.Add(f'Satellite/{i}')
                targetCoverage.UseObjectTimes = True
                targetCoverage.Compute()

                targetCoverageFOM = targetCoverage.FOM
                targetCoverageFOM.SetDefinitionType(AgEFmDefinitionType.eFmNumberOfAccesses)
                targetCoverageFOM.Definition.SetComputeType(AgEFmCompute.eTotal)
                
                DP = targetCoverage.DataProviders.Item('FOM by Time').Exec(self.scenario.StartTime, self.scenario.StopTime, 15)
                data = DP.DataSets.ToArray()
                data = np.delete(data,2,1)
                fig.add_trace(
                    go.Scatter(x=data[:,0], y=data[:,1]),
                    row=list(self.targets.keys()).index(j)//3+1, col=list(self.targets.keys()).index(j)%3+1
                )
            fig.update_xaxes(showticklabels=False) # hide all the xticks
            fig.show()