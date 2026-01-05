// Import necessary libraries.
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <cctype>    
#include <sstream> 

// Categorise each cell in the grid according to the different cell types. 
#define EMPTY   0
#define TREE    1
#define BURNING 2
#define DEAD    3

// Distribute each row of the grid between processes
void distribute_grid(int N, int iproc, int nproc, int &i0, int &i1) {
    // Each process owns a selection of rows based on its rank. 
    i0 = iproc * (N / nproc);
    i1 = (iproc + 1) * (N / nproc);

    // The last process gets spare rows. 
    if (iproc == nproc - 1) {
        i1 = N;
    }
}

// Read example grid (input_grid.txt) from file and returns a 1D vector. 
std::vector<int> read_grid(std::string filename, int &Ni, int &Nj) {
    std::ifstream infile(filename);


    // Error handling incase file is not found. 
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::string line;
    std::vector< std::vector<int> > temp_grid;

    // Read in the file line by line, skipping empty lines. 
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<int> row;
        int num;
        while (iss >> num) {
            row.push_back(num);
        }
        if (!row.empty()) {
            temp_grid.push_back(row);
        }
    }
    infile.close();

    // Sets the number of rows and columns. 
    Ni = temp_grid.size();
    if (Ni > 0) {
        Nj = temp_grid[0].size();
    } else {
        Nj = 0;
    }

    // Convert 2D vector to a 1D vector.
    std::vector<int> grid;
    grid.reserve(Ni * Nj);
    for (const auto &row : temp_grid) {
        assert(row.size() == Nj && "Inconsistent number of columns in grid file.");
        for (int val : row) {
            grid.push_back(val);
        }
    }
    return grid;
}

// Write grid to file in the same format.
void write_grid(std::string filename, const std::vector<int> &grid, int Ni, int Nj) {
    std::ofstream outfile(filename);
    for (int i = 0; i < Ni; i++) {
        for (int j = 0; j < Nj; j++) {
            outfile << grid[i * Nj + j] << " ";
        }
        outfile << "\n";
    }
    outfile.close();
}

// Generate a random grid of size Ni x Nj.
// Each cell becomes TREE with probability p.
std::vector<int> random_grid(int Ni, int Nj, double p) {
    std::vector<int> grid(Ni * Nj, EMPTY);
    for (int i = 0; i < Ni; i++) {
        for (int j = 0; j < Nj; j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < p) {
                grid[i * Nj + j] = TREE;
            }
        }
    }
    return grid;
}

// Updates the grid for one time step, according the the forest fire rules laid out
// in project instructions.  
void update_fire(std::vector<int> &local_grid, std::vector<int> &local_new, int local_rows, int Nj,
                 const std::vector<int> &top_row, const std::vector<int> &bottom_row) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < Nj; j++) {
            int index = i * Nj + j;
            int current = local_grid[index];
            int new_state = current;

            // If the cell is a tree, check to see if it should catch fire. 
            if (current == TREE) {
                bool neighbour_burning = false;
                // Check the cell above. 
                if (i == 0) {
                    if (!top_row.empty() && top_row[j] == BURNING)
                        neighbour_burning = true;
                } else {
                    if (local_grid[(i - 1) * Nj + j] == BURNING)
                        neighbour_burning = true;
                }
                // Check the cell below. 
                if (i == local_rows - 1) {
                    if (!bottom_row.empty() && bottom_row[j] == BURNING)
                        neighbour_burning = true;
                } else {
                    if (local_grid[(i + 1) * Nj + j] == BURNING)
                        neighbour_burning = true;
                }
                // Check the cell to the left
                if (j > 0 && local_grid[i * Nj + (j - 1)] == BURNING)
                    neighbour_burning = true;
                // Check the cell to the right
                if (j < Nj - 1 && local_grid[i * Nj + (j + 1)] == BURNING)
                    neighbour_burning = true;

                // If any neighbour in burning, the tree catches fire. 
                if (neighbour_burning)
                    new_state = BURNING;

            // If the tree is one fire, it becomes burnt and dies in the next step. 
            } else if (current == BURNING) {
                new_state = DEAD;
            }
            local_new[index] = new_state;
        }
    }
}

int main(int argc, char **argv) {
    // Initialise the MPI me. 
    MPI_Init(&argc, &argv);

    // Get the total number of processes and the process's rank
    int nproc, iproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    int Ni, Nj;
    std::vector<int> global_grid;
    double p = 0.0;
    bool file_mode = false, random_mode = false, multi_run = false;
    int M = 1;
    std::string grid_filename;

    // Check to see how the user of the programme wishes to run the programme. 
    // If first argument is not a number (input_grid.txt), run in file input mode. 
    if (argc >= 2 && !std::isdigit(argv[1][0])) {
        file_mode = true;
        grid_filename = argv[1];
    } else if (argc >= 4) {
        
        // If it is a number, use a randomly generated grid with dimensions Ni, Nj,
        // and probability p. 
        random_mode = true;
        Ni = std::atoi(argv[1]);
        Nj = std::atoi(argv[2]);
        p = std::atof(argv[3]);

        // Allows user to choose how many times they wish to repeat the simulation. 
        if (argc >= 5) {
            multi_run = true;
            M = std::atoi(argv[4]);
        }
    } else {

        // Error handling, showing how to use the programme. 
        if (iproc == 0) {
            std::cout << "Usage:" << std::endl;
            std::cout << "  File mode: " << argv[0] << " filename" << std::endl;
            std::cout << "  Random mode: " << argv[0] << " Ni Nj p [M]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    double total_steps = 0.0;
    int total_bottom = 0;
    double total_time = 0.0;

    // If the user wishes to repeat the simulation, run the simulation that many times
    for (int run = 0; run < M; run++) {
        if (file_mode) {
            if (iproc == 0) {
                // Read the grid from file.
                global_grid = read_grid(grid_filename, Ni, Nj);
                // Set trees in the top row of the grid on fire for file input.  
                for (int j = 0; j < Nj; j++) {
                    if (global_grid[j] == TREE)
                        global_grid[j] = BURNING;
                }
            }
        } else if (random_mode) {
            if (iproc == 0) {
                srand(time(NULL) + run);
                global_grid = random_grid(Ni, Nj, p);
                // Set trees in the top row of the grid on fire for random grid. 
                for (int j = 0; j < Nj; j++) {
                    if (global_grid[j] == TREE)
                        global_grid[j] = BURNING;
                }
            }
        }
        // Share grid size with all processes
        MPI_Bcast(&Ni, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Nj, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize the grid on other processes so that they can receive the 
        // full grid
        if (iproc != 0) {
            global_grid.resize(Ni * Nj);
        }
        // Share grid data with all processes.
        MPI_Bcast(global_grid.data(), Ni * Nj, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate which rows this process will handle. 
        int i0, i1;
        distribute_grid(Ni, iproc, nproc, i0, i1);
        int local_rows = i1 - i0;

        // Copy just the portion of the grid this process requires. 
        std::vector<int> local_grid(local_rows * Nj, EMPTY);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < Nj; j++) {
                local_grid[(i - i0) * Nj + j] = global_grid[i * Nj + j];
            }
        }

        int steps = 0;
        bool global_burning = true;
        bool bottom_reached = false;

        // Start timer of the simulation
        double start_run = MPI_Wtime();

        // Continue to update the forest fire until no burning cells remain
        while (global_burning) {
            std::vector<int> top_row(Nj, EMPTY);
            std::vector<int> bottom_row(Nj, EMPTY);

            // Exchange top and bottom rows with neighbouring processes
            if (iproc > 0) {
                MPI_Send(&local_grid[0], Nj, MPI_INT, iproc - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(top_row.data(), Nj, MPI_INT, iproc - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (iproc < nproc - 1) {
                MPI_Send(&local_grid[(local_rows - 1) * Nj], Nj, MPI_INT, iproc + 1, 1, MPI_COMM_WORLD);
                MPI_Recv(bottom_row.data(), Nj, MPI_INT, iproc + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Create a new grid to store the updated state
            std::vector<int> local_new(local_rows * Nj, EMPTY);
            update_fire(local_grid, local_new, local_rows, Nj, top_row, bottom_row);
            
            // Replace old grid with new grid
            local_grid = local_new;
            steps++;

            // Check if any cells are still burning
            bool local_burning = false;
            for (int i = 0; i < local_rows * Nj; i++) {
                if (local_grid[i] == BURNING) {
                    local_burning = true;
                    break;
                }
            }

            // Check if fire has reached the bottom row. 
            bool local_bottom = false;
            if (iproc == nproc - 1 && local_rows > 0) {
                for (int j = 0; j < Nj; j++) {
                    if (local_grid[(local_rows - 1) * Nj + j] == BURNING) {
                        local_bottom = true;
                        break;
                    }
                }
            }

            // Share information with all processes
            int i_burning = local_burning ? 1 : 0;
            int i_bottom = local_bottom ? 1 : 0;
            int global_flag = 0, bottom_flag = 0;
            MPI_Allreduce(&i_burning, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            MPI_Allreduce(&i_bottom, &bottom_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            
            // Update loop condition
            global_burning = (global_flag == 1);
            if (bottom_flag == 1)
                bottom_reached = true;
        }
        // End simulation timer
        double end_run = MPI_Wtime();

        // Save results for the user to see. 
        double sim_time = end_run - start_run;
        total_steps += steps;
        if (bottom_reached)
            total_bottom++;
        total_time += sim_time;
    }

    // Only rank 0 prints the final results. 
    if (iproc == 0) {
        std::cout << "\n\nStatistics for " << M << " runs on a " << Ni << "x" << Nj
                  << " grid with probability " << p << ":\n";
        std::cout << "Average simulation steps: " << total_steps / M << "\n";
        std::cout << "Fire reached the bottom in " << (total_bottom * 100.0 / M) << "% of runs.\n";
        std::cout << "Average simulation time: " << total_time / M << " seconds.\n";
        std::cout << "Total simulation time: " << total_time << " seconds.\n\n";
    }

    // End the MPI programme. 
    MPI_Finalize();
    return 0;
}
