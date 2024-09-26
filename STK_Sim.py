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

# Define a class for simulating an STK (Satellite Tool Kit) scenario
class STK_Simulation:
    # Constructor to initialize the STK simulation environment
    def __init__(self,GUI,Filename):
        # Release any existing STK instances to prevent conflicts
        STKDesktop.ReleaseAll()
        # If the GUI argument is True, start STK with a graphical user interface
        if GUI == True:
            # Start new instance of STK with GUI visible
            self.stk = STKDesktop.StartApplication(visible=True) #using optional visible argument
            # Get the IAgStkObjectRoot interface, reguired to control STK
            self.root = self.stk.Root
        else:
            # Start new instance of STK Engine with no GUI
            self.stk = STKEngine.StartApplication(noGraphics=True) # optionally, noGraphics = True
            # Get the IAgStkObjectRoot interface for interacting with the STK Engine
            self.root = self.stk.NewObjectRoot()
        # Creating a new scenario in STK using the provided Filename (scenario name)
        self.scenario = self.root.NewScenario(Filename)
        # Set default time step for the simulation (in seconds)
        self.dt = 60 # time step of 60 seconds
        # Set a flag for whether interpolation is used (default is False)
        self.Interpolate = False
        # Enable parallel computing in STK for improved performance
        self.root.ExecuteCommand("Parallel / AutomaticallyComputeInParallel On")
        # self.root.ExecuteCommand(f"Parallel / Configuration ParallelType Local NumberOfLocalCores {os.cpu_count()}")

    # Method to load target data from a file
    def Target_Loader(self,Filename):
        # Initialize lists to store target-related data (e.g., bins, times)
        self.targets = []
        self.target_bins = []
        self.target_times = []
        # Unload any previously loaded target objects in the current scenario to start fresh
        for target in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eTarget):
            target.Unload()
        # Read target data from a CSV file using pandas
        # The file is assumed to be delimited by commas
        data = pd.read_csv(Filename,delimiter=',')

        # Loop over each row in the target data (each target)
        for target_num in range(len(data)):
            # Create a new target object in the STK scenario and name it "Target_<target_num>"
            # 'AgESTKObjectType.eTarget' is used to specify the type of object being created (a target)
            self.targets.append(self.root.CurrentScenario.Children.New(AgESTKObjectType.eTarget, f"Target_{target_num}"))
            # IAgFacility target: Target Object
            # Assign the geodetic position (latitude, longitude, altitude) for the newly created target
            # The latitude and longitude are taken from the corresponding columns in the 'data' DataFrame
            # Altitude is initially set to 0, meaning the target is placed on the surface of the Earth
            self.targets[-1].Position.AssignGeodetic(float(data['Lat'][target_num]), float(data['Lon'][target_num]), 0)  # Latitude, Longitude, Altitude
            # Set altitude to height of terrain
            self.targets[-1].UseTerrain = True
            # Set altitude to a distance above the ground
            self.targets[-1].HeightAboveGround = 0   # km
            self.target_bins.append(np.zeros([36,9]))
            self.target_times.append(self.root.CurrentScenario.StopTime)

    # Method to load satellite data from a file        
    def Satellite_Loader(self,Filename,External_Pointing_File=False):
        # Initialize lists to store satellite and radar objects
        self.satellites = []
        self.radars = []

        # Unload any previously loaded satellites in the current scenario to start fresh
        for satellite in self.root.CurrentScenario.Children.GetElements(AgESTKObjectType.eSatellite):
            satellite.Unload()

        # Read satellite data from a CSV file using pandas
        # The file is assumed to be delimited by commas, containing parameters for satellite orbits
        data = pd.read_csv(Filename,delimiter=',')

        # Loop over each row in the satellite data (each satellite)
        for satellite_num in range(len(data)):
            # Create a new satellite object in the STK scenario and name it "Satellite_<satellite_num>"
            # 'AgESTKObjectType.eSatellite' specifies the type of object being created (a satellite)
            self.satellites.append(self.root.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, f"Satellite_{satellite_num}"))

            # IAgSatellite satellite: Satellite object
            # Get the satellite object to modify its orbital parameters
            # 'keplerian' stores the orbital state in the classical (Keplerian) elements form
            keplerian = self.satellites[-1].Propagator.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
            # Specify the type of orbital elements being used (altitude-based size/shape and true anomaly for location)
            keplerian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeAltitude
            keplerian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
            keplerian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN

            # Assign the perigee and apogee altitude values:
            # Assign the perigee and apogee altitudes (the closest and farthest points in the orbit) from the data
            keplerian.SizeShape.PerigeeAltitude = float(data['Per'][satellite_num])      # km
            keplerian.SizeShape.ApogeeAltitude = float(data['Apo'][satellite_num])        # km

            # Assign the other desired orbital parameters:
            # Assign other classical orbital parameters from the data
            keplerian.Orientation.Inclination = float(data['Inc'][satellite_num])               # deg
            keplerian.Orientation.ArgOfPerigee = float(data['AoP'][satellite_num])        # deg
            keplerian.Orientation.AscNode.Value = float(data['Asc'][satellite_num])            # deg
            keplerian.Location.Value = float(data['Loc'][satellite_num])                             # deg

            # Apply the changes made to the satellite's state and propagate:
            # Apply the changes made to the satellite's orbital state (Keplerian parameters)
            self.satellites[-1].Propagator.InitialState.Representation.Assign(keplerian)
            # Propagate the satellite's orbit based on the initial conditions
            self.satellites[-1].Propagator.Propagate()

            # IAgSatellite satellite: Satellite object
            # self.radars[satellite_num] = self.satellites[satellite_num].Children.New(AgESTKObjectType.eRadar, f'Radar{i+1}')
            # self.radars[satellite_num].CommonTasks.SetPatternSimpleConic(5, 0.1)
            # self.radars[satellite_num].CommonTasks.SetPatternSAR(0,90,0,0,data['Per'][i])
            # self.radars[satellite_num].SetPointingType(5)
            # for j in self.targets:
            #     self.radars[satellite_num].Pointing.Targets.Add(f'*/Target/{j}')

    # Reset the target bins and times for all targets
    # This method is used to clear/reset the data collected for each target, preparing them for a new computation.
    def Reset_Target_Bins(self):
        # loop through each target
        for idx in range(len(self.targets)):
            # reset the bin data for the target to a 36x9 zero matrix
            self.target_bins[idx] = np.zeros([36,9])
            # reset the time for each target to the scenario stop time (end of the simulation)
            self.target_times[idx] = self.root.CurrentScenario.StopTime
    # Update the target bins with new data
    # This method updates the bin count for a specific target and bin at a specific time.
    def Update_Target_Bins(self,time,bin,target_number):
       # If the bin for the given target is empty (has a value of 0), set the target time to the current time
        if self.target_bins[target_number][bin//9,bin%9] == 0:
            self.target_times[target_number] = time
        # Increment the bin count for the target at the specified bin location
        # The bin index is split into a 2D matrix of size 36x9, using integer division and modulo
        self.target_bins[target_number][bin//9,bin%9]+=1
        # Return 0 as a default exit status
        return 0

    # Compute azimuth and elevation for all satellite-target pairs
    # This method calculates access (visibility) between satellites and targets, storing the results.
    def Compute_AzEl(self,enable_print=True):
        # Reset target bins before performing calculations to clear any previous data
        self.Reset_Target_Bins()
        # Use a progress bar to monitor the process of computing access for all satellites and targets
        # 'alive_bar' is used to track progress, with one iteration for each satellite-target pair
        with alive_bar(len(self.targets)*len(self.satellites),force_tty=True,bar='classic',title='- Computing_AzEl',length=10,disable=not(enable_print)) as bar:
            # loop through each satellite
            for sat in self.satellites:
                # loop through each target
                for tar_num,tar in enumerate(self.targets):
                    # get access (line-of-sight visibility) from the target to the current satellite
                    access = tar.GetAccessToObject(sat)
                    # Compute the access, which may involve calculating azimuth and elevation angles
                    access.ComputeAccess()
                    # Retrieve access data for Azimuth, Elevation, and Time from STK's AER (Azimuth-Elevation-Range) data provider
                    # The 'ExecElements' method fetches the data over the time range from the scenario's start time to stop time, at intervals defined by 'self.dt'
                    DataSets = access.DataProviders.GetItemByName('AER Data').Group.Item(0).ExecElements(self.root.CurrentScenario.StartTime,
                                                                                                        self.root.CurrentScenario.StopTime,
                                                                                                        self.dt,['Time','Azimuth','Elevation']).DataSets
                    # Loop through the DataSets, where each set contains the 'Time', 'Azimuth', and 'Elevation' data
                    # DataSets are structured so that 'idx' points to 'Time', 'idx+1' points to 'Azimuth', and 'idx+2' points to 'Elevation'
                    for idx in range(0,DataSets.Count,3):
                        # Extract the time, azimuth, and elevation values from the data sets
                        time = DataSets.Item(idx).GetValues() # time vals from the data set 
                        az = DataSets.Item(idx+1).GetValues() # azimuth vals from the data set (degrees)
                        el = np.abs(DataSets.Item(idx+2).GetValues()) # elevation vals from the data set (degrees), using abs val
                        # if interpolation is enabled, interoplate the time, azimuth, and elevation data 
                        if self.Interpolate:
                            time,az,el = Interpolate(time,az,el)
                        # bin the azimuth and elevation data into discrete categories
                        # each bin is computed as (azimuth // 10) * 9 + (elevation // 10) to index a 2D bin structure
                        bins = np.array([(a//10)*9+(e//10) for a,e in zip(az,el)]).astype(int)
                        # loop through each time step and update the target bins with the computed values
                        for j in range(len(time)):
                            # Call the Update_Target_Bins method to record the time and bin information for the current target
                            self.Update_Target_Bins(time[j],bins[j],tar_num)
                    # update the progress bar after processing each satellite-target pair
                    bar()
        # Return 0 to indicate the method has completed (un)successfully?
        return 0
    
    # # Method to retrieve data provider (DP) information for satellites and convert it to a pandas DataFrame
    # 'bus_name' specifies the name of the data provider or data group in the satellite objects
    def Get_Satellite_DP(self,bus_name,enable_print=True):
        # Initialize an empty list to store the data frames for each satellite
        dfs = []
        # Split the 'bus_name' by '/' to handle cases where the data provider is part of a group
        splits = bus_name.split("/")
        # Use a progress bar to track the computation across all satellites
        with alive_bar(len(self.satellites),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10,disable=not(enable_print)) as bar:
            # Loop through each satellite to retrieve data provider information
            for sat in self.satellites:
                # If 'bus_name' is part of a group (contains two parts), access the group first and then the item
                if len(splits) == 2:
                    bus = sat.DataProviders.GetItemByName(splits[0]).Group.GetItemByName(splits[1])
                # Otherwise, directly access the data provider by its name
                elif len(splits) == 1:
                    bus = sat.DataProviders.GetItemByName(bus_name)
                # Execute the data provider to retrieve the data from the scenario's start to stop time with a specified time step (self.dt)
                df = bus.Exec(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt).DataSets.ToPandasDataFrame()
                # Convert the columns of the resulting DataFrame to float (if possible), ignoring errors for non-numeric columns
                for col in df.columns:
                    df[col] = df[col].astype(float,errors='ignore')
                # Append the processed DataFrame to the list of data frames
                dfs.append(df)
                # Update the progress bar after each satellite's data is processed
                bar()
        # Return the list of DataFrames containing data provider information for all satellites
        return dfs
    
    # Method to retrieve access data provider information between two objects (obs1 and obs2) and convert it to a list of data frames
    # 'bus_name' specifies the name of the data provider or data group, and 'Total_Elements' specifies if only certain elements should be retrieved
    def Get_Access_DP(self,obs1,obs2,bus_name,Total_Elements=False,enable_print=True):
        dfs = [] # initialize an empty list to store the data frames for each access computation
        splits = bus_name.split("/") # split the 'bus_name' by '/' to handle ground-based data providers
        # use a progress bar to track the computation across all satellite-target pairs
        with alive_bar(len(self.satellites)*len(self.targets),force_tty=True,bar='classic',title=f'- Computing_{bus_name}',length=10,disable=not(enable_print)) as bar:
            # loop through the first set of objects (obs1)
            for ob1 in obs1:
                dfs.append([]) # initialize an empty list to store results for each 'obs1' object
                # loop through the 2nd set of objects (obs2)
                for ob2 in obs2:
                    # compute access between the 2 objects (ob1 and ob2)
                    access = ob1.GetAccessToObject(ob2)
                    access.ComputeAccess()
                    # access the data provider either directly or from a group, depending on 'bus_name'
                    if len(splits) == 2:
                        bus = access.DataProviders.GetItemByName(splits[0]).Group.GetItemByName(splits[1])
                    if len(splits) == 1:
                        bus = access.DataProviders.GetItemByName(splits[0])
                    df = {} # dictionary to store the data for this particular access
                    # if no specific elements are requested, retrieve the full data set
                    if not(Total_Elements):
                        DataSets = bus.Exec(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt).DataSets
                        Elements = DataSets.ElementNames # get the names of the data elements (e.g., Time, Azimuth, Elevation)
                    else:
                        # if specific elements are requested, retrieve only those
                        DataSets = bus.ExecElements(self.root.CurrentScenario.StartTime,self.root.CurrentScenario.StopTime,self.dt,Total_Elements).DataSets
                        Elements = Total_Elements
                    # if there are data elements available
                    if len(Elements) > 0:
                        # initialize an empty list for each element
                        for e in Elements:
                            df[e] = []
                        # loop through the data sets and extract the values for each element    
                        for idx in range(0,DataSets.Count,len(Elements)):
                            for e_idx,e in enumerate(Elements):
                                df[e].extend(DataSets.Item(idx+e_idx).GetValues()) # extend the list with the element values
                        # append the access data for the current object pair to the list
                        dfs[-1].append(df)
                    # update the progress bar after each object pair is processed
                    bar()
        # Return the list of dictionaries containing access data for all object pairs
        return dfs
    # Method to update the mass properties (mass and inertia matrix) of the satellites
    def Update_Mass_Properties(self,M=250,I=[[288,0,0],
                                             [0,88.88,0],
                                             [0,0,288]]):
        # loop through each satellite in the scenario and set its mass and inertia matrix
        for sat in self.satellites:
            sat.MassProperties.Mass = M # set the mass of the satellite
            # set the components of the inertia matrix (Ixx, Ixz, Iyy, Iyz, Izz)
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