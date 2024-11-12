import random
import numpy as np

matrix = []
trucks = {}
INT_MAX = float('inf')

def read_input_file(filename):
    """
    Read and parse input file
    Inputs -> filename: Path to input file containing matrix and truck data
    Outputs -> None, updates global matrix and trucks variables
    """
    global matrix, trucks
    print(f"Starting to read file: {filename}")
    
    with open(filename, 'r') as file:
        print("File opened successfully")
        lines = file.readlines()
        print(f"Read {len(lines)} lines from file")
        
    # Parse distance matrix
    print("\nParsing distance matrix...")
    matrix = []
    for i, line in enumerate(lines):
        if '#' in line:  # This indicates we've reached truck information
            print(f"Found truck information delimiter at line {i+1}")
            break
        row = [int(x.strip()) if x.strip() != 'N' else 'N' for x in line.strip().split(',')]
        matrix.append(row)
    print(f"Parsed distance matrix: {len(matrix)}x{len(matrix[0])} dimensions")
    
    # Parse truck information
    print("\nParsing truck information...")
    trucks = {}
    truck_lines = lines[len(matrix):]
    for i, line in enumerate(truck_lines):
        if line.strip():
            truck_info = line.strip().split('#')
            trucks[truck_info[0]] = int(truck_info[1])
            print(f"Added truck {truck_info[0]} with capacity {truck_info[1]}")
    
    print(f"\nFinished parsing. Found {len(matrix)} locations and {len(trucks)} trucks")

def printGrid():
    """
    Print the distance matrix in a formatted grid layout
    Inputs -> None, uses global matrix variable
    Outputs -> None, prints formatted grid to console
    """
    for row in matrix:
        formatted_row = []
        for x in row:
            if x == 'N':
                formatted_row.append(' N ')  # Show N for no path
            else:
                formatted_row.append(f'{float(x):3.0f}')  # Show weights as numbers, rounded to integer
        print(' '.join(formatted_row))  # Print row with spaces between cells
        
def printTracks():
    """
    Print all possible tracks between locations using letter notation
    Inputs -> None, uses global matrix variable
    Outputs -> None, prints all possible connections to console
    """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, row in enumerate(matrix):
        track = letters[i] + ': '  # Start with location letter
        connections = []
        for j, val in enumerate(row):
            if val != 'N' and i != j:  # Skip self-connections and no paths
                connections.append(f"{letters[i]}->{letters[j]}({val})")
        if connections:
            track += ', '.join(connections)  # Join all connections with commas
        else:
            track += 'No connections'  # Show if location has no valid connections
        print(track)
    
def printTrucks():
    """
    Print truck information showing ID and capacity
    Inputs -> None, uses global trucks variable
    Outputs -> None, prints truck details to console
    """
    for truck_id, capacity in trucks.items():
        print(f"{truck_id}: {capacity}")

def write_output_file(filename, trucks, total_distance):
    """
    Write the solution routes and total distance to output file
    Inputs -> filename: Name of output file to write to
              trucks: Dictionary mapping truck IDs to their routes
              total_distance: Total distance of all routes combined
    Outputs -> None, writes solution to specified file
    """
    with open(filename, 'w') as file:
        # Convert numeric indices to letters (0=a, 1=b, etc.)
        for truck_id, truck_obj in trucks.items():
            route_letters = [chr(ord('a') + loc) for loc in truck_obj.route]  # Convert each index to corresponding letter
            file.write(f"{truck_id}#{','.join(route_letters)}\n")  # Write truck route in required format
        file.write(str(total_distance))  # Write total distance on last line

class truck:
    def __init__(self, name, capacity):
        """
        Initialize a new truck object
        Inputs -> name: Unique identifier for the truck
                 capacity: Maximum number of locations this truck can visit
        Outputs -> None, initializes truck attributes
        """
        self.name = name
        self.capacity = capacity
        self.route = []
        self.current_location = 0
        self.delivery_count = 0
        self.next_set = [1,2,3,4,5]
        print(f"Created truck {name} with capacity {capacity}")
    
    def isCapable(self):
        """
        Check if truck has remaining capacity to visit more locations
        Inputs -> None
        Outputs -> Boolean indicating if truck can take more locations
                  True if remaining capacity > 0, False otherwise
        """
        remaining = self.capacity - self.delivery_count
        print(f"Truck {self.name} has {remaining} remaining capacity")
        return remaining > 0
    
    def move(self, new_location):
        """
        Move truck to new location and update its route and available locations
        Inputs -> new_location: Index of location to move truck to
        Outputs -> None, updates truck's current location, route and next_set
        """
        print(f"Moving truck {self.name} from {self.current_location} to {new_location}")
        self.current_location = new_location
        self.route.append(new_location)
        self.next_set.remove(new_location)
        self.delivery_count += 1
    def make_recovery_move(self, new_location):
        """
        Move truck to a new location as a recovery move
        Inputs -> new_location: Index of location to move truck to
        Outputs -> None, updates truck's current location, route and next_set
        """
        print(f"Making recovery move for truck {self.name} to {new_location}")
        self.current_location = new_location
        self.route.append(new_location)
    
    def get_notification(self,others_best_move):
        """
        Remove a location that another truck has claimed from available locations
        Inputs -> others_best_move: Location index that another truck has taken
        Outputs -> None, updates next_set by removing claimed location
        """
        print(f"Truck {self.name} removing {others_best_move} from available locations")
        self.next_set.remove(others_best_move)

class grid:
    def __init__(self,matrix):
        """
        Initialize grid with distance matrix
        Inputs -> matrix: 2D list/array containing distances between locations
        Outputs -> None, initializes grid with distance matrix
        """
        self.matrix = matrix    
        print("Created grid from distance matrix")
    
    def get_distance(self, x, y):
        """
        Get distance between two locations in the grid
        Inputs -> x: Source location index
                 y: Destination location index
        Outputs -> Integer distance between locations if path exists
                  INT_MAX if no path exists between locations
        """
        if self.matrix[x][y] == 'N':
            print(f"No path exists between {x} and {y}")
            return INT_MAX
        else:
            print(f"Distance between {x} and {y}: {self.matrix[x][y]}")
            return self.matrix[x][y]
    
class hill_climber:
    def __init__(self, grid):
        """
        Initialize hill climbing optimizer
        Inputs -> grid: Grid object containing distance matrix
        Outputs -> None, initializes optimizer with grid and zero total distance
        """
        self.grid = grid
        self.total_distance = 0
        self.covered_deliveries = 0
        print("Created hill climber optimizer")
    
    def find_local_max(self):
        """
        Find local maximum solution by iteratively assigning locations to trucks
        Inputs -> None, uses global trucks dictionary
        Outputs -> Integer total distance of final solution after all assignments
        """
        global trucks
        total_capacity = sum(truck.capacity for truck in trucks.values())
        print(f"Starting optimization with total capacity: {total_capacity}")
        
        while total_capacity > 0 and self.covered_deliveries < len(self.grid.matrix):
            random_truck = random.choice(list(trucks.values()))
            print(f"\nSelected random truck: {random_truck.name} with current location {random_truck.current_location} current total distance {self.total_distance}")
            
            if random_truck.isCapable():
                print("Truck has remaining capacity")
                best_move, add_distance, status = self.get_best_move(random_truck)
                if status == 0:
                    print("Successfully assigned location")
                    total_capacity -= 1
                    self.total_distance += add_distance
                    self.covered_deliveries += 1
                elif status == -1:
                    print("Successfully assigned(recovery) location")
                    self.total_distance += add_distance
                elif status == -2:
                    print("Could not find valid move")
            else:
                print("Truck is at capacity")
                
        print(f"Finished optimization with total distance: {self.total_distance}")
        return self.total_distance

    def get_best_move(self, truck):
        """
        Find optimal next location for given truck based on minimum distance
        Inputs -> truck: Truck object to find next move for
        Outputs -> Integer minimum distance to best next location
                  -1 if no valid moves are available
        """
        print(f"\nFinding best move for truck {truck.name}")
        temp_next = truck.next_set
        print(f"Available locations: {temp_next}")
        
        temp_weights = [self.grid.get_distance(truck.current_location, i) for i in temp_next]
        print(f"Distances to available locations: {temp_weights}")
        
        if min(temp_weights) == INT_MAX:
            print("No valid moves found, finding recovery move")
            recovery_move, recovery_distance, status = self.get_recovery_move(truck)
            if status == -1:
                return 0, INT_MAX, -2
            print(f"Recovery move is to location {recovery_move}")
            best_move = recovery_move

            truck.make_recovery_move(best_move)

            return best_move, recovery_distance, -1
        else:   
            best_move = temp_next[temp_weights.index(min(temp_weights))]
            print(f"Best move is to location {best_move}")
            
            truck.move(best_move)
            
            # Notify other trucks
            for t in trucks.values():
                if t != truck:
                    t.get_notification(best_move)
                
            return best_move, min(temp_weights), 0
    
    def get_recovery_move(self, truck):
        """
        Find recovery move for given truck by selecting a location not in next_set
        Inputs -> truck: Truck object to find recovery move for
        Outputs -> Integer index of recovery move location
        """
        print("[Recovery] Finding all available moves...")
        available_moves = [i for i in range(len(self.grid.matrix)) if self.grid.get_distance(truck.current_location, i) != float('inf')]
        print(f"[Recovery] Initial available moves: {available_moves}")
        
        # Remove current location from available moves
        available_moves = [i for i in available_moves if i != truck.current_location]
        print(f"[Recovery] Available moves after removing current location: {available_moves}")
        
        if not available_moves:
            print("[Recovery] No available moves found")
            return 0, INT_MAX, -1
            
        recovery_move = random.choice(available_moves)
        print(f"[Recovery] Initially selected recovery move: {recovery_move}")
        
        while recovery_move in truck.next_set:
            print(f"[Recovery] Move {recovery_move} is in next_set, removing and trying again")
            available_moves.remove(recovery_move)
            if not available_moves:
                print("[Recovery] No more available moves after filtering")
                return 0, INT_MAX, -1
            recovery_move = random.choice(available_moves)
            print(f"[Recovery] New selected recovery move: {recovery_move}")
            
        distance = self.grid.get_distance(truck.current_location, recovery_move)
        print(f"[Recovery] Recovery move is to location {recovery_move} with distance {distance}")
        return recovery_move, distance, 0

def print_routes(trucks):
    """
    Print formatted routes for all trucks using letter notation
    Inputs -> trucks: Dictionary mapping truck IDs to truck objects
    Outputs -> None, prints each truck's route with locations as letters
    """
    location_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
    print("\nTruck Routes:")
    print("-" * 30)
    for t in trucks.values():
        route = [location_map.get(loc, str(loc)) for loc in t.route]
        print(f"Truck {t.name}: A -> {' -> '.join(route)}")
    print("-" * 30)

def main():
    read_input_file('input.txt')
    print("Matrix:")
    printGrid()
    print('*'*10+"\n")
    print("Tracks:")
    printTracks()
    print('*'*10+"\n")
    print("Trucks:")
    printTrucks()

    # Number of times to run the optimization
    trials = 50
    # Track the best distance found across all trials
    best_total_distance = INT_MAX
    # Store the best truck configuration found
    best_trucks = {}

    # Create new grid instance for this trial
    my_grid = grid(matrix)

    # Initialize trucks with their capacities
    for truck_id, capacity in trucks.items():
        trucks[truck_id] = truck(truck_id, capacity)
    
    # Run multiple trials to find best solution
    for i in range(trials):
        # Create hill climber optimizer with current grid
        my_hill_climber = hill_climber(my_grid)

        # Run optimization to find local maximum
        total_distance = my_hill_climber.find_local_max()

        # Print routes for all trucks in current solution
        print_routes(trucks)

        # Update best solution if current is better
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_trucks = {}
            for k, t in trucks.items():
                best_trucks[k] = truck(t.name, t.capacity)
                best_trucks[k].route = t.route.copy()

        # Reset trucks for next trial
        for t in trucks.values():
            t.route = []
            t.current_location = 0
            t.next_set = [1,2,3,4,5]
            t.delivery_count = 0

        print_routes(best_trucks)
    
    # Print routes for best solution
    print("\n" + "="*20 + "\nBest Solution Found:\n" + "="*20)
    print_routes(best_trucks)
    print(f"Total distance: {best_total_distance}")
    
    # Write the best solution found to output file
    write_output_file('output.txt', best_trucks, best_total_distance)

if __name__ == "__main__":
    main()