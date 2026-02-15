import io
import time
import random
import hashlib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from streamlit_javascript import st_javascript
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis
from functools import wraps
from tqdm import tqdm

class TModel:
    """
    Class for a cellular automata, modeling tumor growth.

    Parameters:
        cycles (int): duration of the model given in hours
        side (int): the length of the side of field (10um)
        pmax (int): maximum proliferation potential of RTC
        PA (int): chance for apoptosis of RTC (in percent)
        CCT (int): cell cycle time of cells given in hours
        Dt (float): time step of the model given in days
        PS (int): STC-STC division chance (in percent)
        mu (int): migration capacity of cancer cells
        I (int): strength of the immune cells (1-5)
        M (int): tumor mutation chance (in percent)
    """
    
    def __init__(self, cycles, side, pmax, PA, CCT, Dt, PS, mu, I, M):
        # Parameters     
        self.cycles = cycles
        self.side   = side
        self.pmax   = pmax
        self.CCT    = CCT
        self.Dt     = Dt
        self.mu     = mu
        self.I      = I
        self.M      = M
        
        # Single model data
        self.stc_number = []
        self.rtc_number = []
        self.wbc_number = []
        self.cancer = []
        self.immune = []
        self.mutate = []
        self.mutmap = []
        self.images = []
        
        # Multiple models data
        self.stats = []
        self.runs  = []
        
        # Chances
        self.PP = 24 * Dt/CCT * 100
        self.PM = 100 * mu/24
        self.PA = PA
        self.PS = PS
        
        # Immune Data
        self.it_ratio = []
        self.kill_day = []
    
    # ---------------------------------------------------------------------
    def init_state(self):
        """
        Creates the initial state with one STC in the middle.
        Creates the field for immune cells and mutations too.
        """

        self.cancer = np.zeros((self.side, self.side))
        self.immune = np.zeros((self.side, self.side))
        self.mutate = np.zeros((self.side, self.side))
        self.mutmap = np.zeros((self.side, self.side))
        
        self.mod_cell(self.side//2, self.side//2, self.pmax+1)
        
    # ---------------------------------------------------------------------
    def find_tumor_cells(self):
        """
        Saves the coordinates of tumor cells to self.tumor_cells.
        """
     
        # Tumor cell coords randomized
        coords = np.argwhere(self.cancer > 0)
        np.random.shuffle(coords)
        self.tumor_cells = coords

    # ---------------------------------------------------------------------
    def count_tumor_cells(self):
        """
        Saves the number of STCs/RTCs to self.stc_number/self.rtc_number.
        """
        
        # Count RTC and STC
        stc_count = np.count_nonzero(self.cancer == self.pmax + 1)
        rtc_count = len(self.tumor_cells) - stc_count
        
        # Save the current number
        self.stc_number.append(stc_count)
        self.rtc_number.append(rtc_count)

    # ---------------------------------------------------------------------
    def get_neighbours(self, x, y, neighbour_type):
        """
        Returns the neighboring coordinates of a given cell in a 2D NumPy matrix.

        Parameters:
            x, y (int): representing the coordinates of the cell
            neighbour_type (int): type of neighboring cells (1-5)
            
        Returns:
            list: a list with the coords of the neighbouring cells
        """

        r_start = max(1, x - 1)
        r_end   = min(self.side - 1, x + 2)
        c_start = max(1, y - 1)
        c_end   = min(self.side - 1, y + 2)
    
        # Extract views of the field and immune grids
        f_view = self.cancer[r_start:r_end, c_start:c_end]
        i_view = self.immune[r_start:r_end, c_start:c_end]
    
        match neighbour_type:
            case 1:  # Empty
                mask = (f_view == 0) & (i_view == 0)
            case 2:  # Tumor
                mask = f_view > 0
            case 3:  # Immune
                mask = i_view > 0
            case 4:  # Any Cell
                mask = (f_view > 0) | (i_view > 0)
            case 5:  # Not Immune
                mask = i_view == 0
    
        matches = np.argwhere(mask) 
        matches += [r_start, c_start]
        
        is_center = (matches[:, 0] == x) & (matches[:, 1] == y)
        return matches[~is_center].tolist()
    
    # ---------------------------------------------------------------------
    def cell_step(self, x, y, step_type):
        """
        The function that makes a single cell do one of the following actions:
        prolif STC - STC, prolif STC - RTC, prolif RTC - RTC, migration (1-4).
        New mutations can appear every time a cell proliferates with M chance.

        Parameters:
            x, y (int): representing the coordinates of the cell
            step_type (int): type of division or migration (1-4)
        """
        
        # Choose random target position
        free_nb = self.get_neighbours(x, y, 1)
        nx, ny = free_nb[random.randint(1,len(free_nb)) - 1]
        
        match step_type:
            case 1:
                # Proliferation STC -> STC + STC
                self.cancer[nx, ny] = self.pmax+1
            case 2:
                # Proliferation STC -> STC + RTC
                self.cancer[nx, ny] = self.pmax
            case 3:
                # Proliferation RTC -> RTC + RTC
                self.cancer[x, y]   -= 1
                self.cancer[nx, ny] = self.cancer[x, y]
            case 4:
                # Migration
                self.cancer[nx, ny] = self.cancer[x, y]
                self.cancer[x, y]   = 0
        
        if step_type < 4 and self.cancer[x, y] == 0:
            self.mutate[x, y] = 0
            
        elif step_type < 4:
            # Inherit mother's mutation
            self.mutate[nx, ny] = self.mutate[x, y]
            
            # Chance of a new mutation
            if self.M >= random.randint(1, 100):
                mut = random.choice([-1,1])
                self.mutate[nx, ny] = np.clip(self.mutate[nx, ny]+mut, -3, 3)
                
                # Mutation influences pp value
                if step_type != 1:
                    self.cancer[nx, ny] = np.clip(self.cancer[nx, ny]+mut, 1, self.pmax)
                    
            self.mutmap[nx, ny] = self.mutate[nx, ny]
        else:
            self.mutate[nx, ny] = self.mutate[x, y]
            self.mutmap[nx, ny] = self.mutate[x, y]
            self.mutate[x,   y] = 0
        
    # ---------------------------------------------------------------------
    def tumor_action(self):
        """
        This is the function that decides what action a cell will do.
        Either kills the cell or calls the 'cell_step' function.
        This function goes through every single cell in the field.
        """
        
        for cell in self.tumor_cells:
            x, y = cell
            is_stc = (self.cancer[x, y] == self.pmax + 1)
        
            # Probabilities
            probs = np.array([self.PA, self.PP, self.PM, 0], dtype=float)
            if is_stc:
                probs[0] = 0
            if not self.get_neighbours(x, y, 1):
                probs[1:3] = 0
            probs = self.mutate_probs(probs, x, y)
            probs /= probs.sum()
        
            # Choose action
            choice = np.random.choice(4, p=probs)
        
            if choice == 0:    # apoptosis
                self.cancer[x, y] = self.mutate[x, y] = 0
                
            elif choice == 1:  # proliferation
                if is_stc and np.random.rand() < self.PS/100:
                    self.cell_step(x, y, 1)   # STC-STC division
                elif is_stc:
                    self.cell_step(x, y, 2)   # STC-RTC division
                else:
                    self.cell_step(x, y, 3)   # RTC-RTC division  
                    
            elif choice == 2:  # migration
                self.cell_step(x, y, 4)

    # ---------------------------------------------------------------------
    def mutate_probs(self, chances, x, y):
        """
        The function that changes the cell action chances
        based on the current mutation status of the cell.
        
        Parameters:
            chances (list of float): the base action chances
            x, y (int): representing coordinates of the cell
        """
        
        mut_state = self.mutate[x, y]/2
        
        if mut_state > 0:
            mut_state += 1
            chances[0] = chances[0]/mut_state        # Decreased chance for apoptosis 
            chances[1] = chances[1]*mut_state        # Increased proliferation chance
        elif mut_state < 0:
            mut_state -= 1
            chances[0] = chances[0]*abs(mut_state)   # Increased chance for apoptosis 
            chances[1] = chances[1]/abs(mut_state)   # Decreased proliferation chance
            
        if chances.sum() <= 100:
            chances[3] = 100 - chances.sum()
        return chances

    # ---------------------------------------------------------------------
    def immune_response(self, offset = 10, alpha = 0.002, it_targ = 0.1, infil = 0.3):
        """
        The function that simulates immune cells.
        Spawns, moves and activates immune cells.
        
        Parameters:
            offset (int): distance of spawnpoints ("frame") from the tumor
            alpha (float): controls strength (slope) of immune exhaustion
            it_targ (float): desired mean immune/tumor ratio during simulation
            infil (float): "searching/infiltrating" threshold for wbcs (0-1)
        """

        # Current tumor cell locations
        self.find_tumor_cells()
        tumor_size = len(self.tumor_cells)
        
        if tumor_size == 0:
            self.immune = np.maximum(0, self.immune - 1)
            self.wbc_number.append(np.count_nonzero(self.immune))
            return

        # Immune spawnpoints = "frame" around tumor
        min_coords = self.tumor_cells.min(axis=0) - offset
        max_coords = self.tumor_cells.max(axis=0) + offset
        
        x1, y1 = np.clip(min_coords, 1, self.side - 2)
        x2, y2 = np.clip(max_coords, 1, self.side - 2)
        
        t = np.column_stack((np.full(y2-y1+1, x1), np.arange(y1, y2+1)))
        b = np.column_stack((np.full(y2-y1+1, x2), np.arange(y1, y2+1)))
        l = np.column_stack((np.arange(x1+1, x2), np.full(x2-x1-1, y1)))
        r = np.column_stack((np.arange(x1+1, x2), np.full(x2-x1-1, y2)))
        
        self.spawnpoints = np.concatenate([t, b, l, r])
        
        # Immune exhaustion = time-dependent decline
        IE = max(1.0 / (1.0 + alpha * self.cycles), 0.2)

        # Saturating spawn (sigmoid-like), delayed onset
        spawn = self.I * (tumor_size / (tumor_size + self.I * 100)) * IE
        
        current_wbc_count = np.count_nonzero(self.immune)
        it_ratio = current_wbc_count / tumor_size
        
        if it_ratio <= it_targ:
            # Choose all spawnpoints
            spawn_mask = np.random.random(len(self.spawnpoints)) < (spawn / 50)
            potential_spawns = self.spawnpoints[spawn_mask]
            
            if len(potential_spawns) > 0:
                # Filter for empty slots
                px, py = potential_spawns[:, 0], potential_spawns[:, 1]
                valid = (self.cancer[px, py] == 0) & (self.immune[px, py] == 0)
                final_coords = potential_spawns[valid]
                
                if len(final_coords) > 0:
                    min_life, max_life = min(24, (self.I-1)*168), (self.I+1)*168
                    self.immune[final_coords[:, 0], final_coords[:, 1]] = np.random.randint(
                        min_life, max_life, size=len(final_coords))
                  
        # Chemoattractant map for tumor density
        self.chemo = (self.cancer > 0).astype(float)
        self.chemo = gaussian_filter(self.chemo, sigma=5)
        self.chemo = self.chemo / np.max(self.chemo)
                
        # Immune action
        coords = np.argwhere(self.immune > 0)
        kills_per_hour = 0
        self.immune_cells = coords
        
        # Temporary immune grid
        new_immune = np.zeros_like(self.immune)

        for (x, y) in self.immune_cells:
            strength = self.immune[x, y]
            if strength <= 0:
                continue
        
            # Kill prob on contact: (0.15 - 0.3, if I=5, IE = 0)
            tumor_nb = self.get_neighbours(x, y, 2)
            if tumor_nb:
                tx, ty = random.choice(tumor_nb)
                kill = (0.05*self.I) * np.exp(-0.25*self.mutate[tx,ty]) * IE
                kill = min(kill, 0.3)
                if np.random.rand() < kill:
                    self.cancer[tx, ty] = 0
                    self.mutate[tx, ty] = 0
                    kills_per_hour += 1

            # # Multiple moves/cycle as immune cells are faster
            moves = int(1 + self.I * (1 - self.chemo[x, y]))
            for _ in range(moves):
                free_nb = self.get_neighbours(x, y, 1)
                if not free_nb:
                    strength -= 1
                    break
        
                # Biased movement towards tumor density (chemotaxis)
                t_dens = [self.chemo[i, j] for (i, j) in free_nb]
                if sum(t_dens) > 0:
                    weights = np.array(t_dens) / sum(t_dens)
                    tx, ty = free_nb[np.random.choice(len(free_nb), p=weights)]
                else:
                    tx, ty = random.choice(free_nb)
                x, y = tx, ty
                strength -= 1
                
            if strength > 0:
                new_immune[x, y] = strength
        self.immune = new_immune

        # Save number of immune cells
        immune_size = len(self.immune_cells)
        self.wbc_number.append(immune_size)
        self.it_ratio.append(immune_size / tumor_size)
        
        # Infiltrating immune cells
        wbc_infil = sum(1 for (x,y) in self.immune_cells if self.chemo[x,y] >= infil)
        if immune_size > 0:
            self.kill_day.append(kills_per_hour / max(1, wbc_infil) * 24)

    # ---------------------------------------------------------------------
    def animate(self, mode):
        """
        Creates and returns animation of the growth.
        
        Parameters:
            mode (int): create figure, save frame or display animation. (1-3)
        
        Returns:
            ArtistAnimation: the animation of the growth (optional)
        """
        
        if mode == 1:
            # Create the figure
            self.fig, self.ax = plt.subplots()
            self.ax.imshow(self.cancer)
            self.ax.set_title(str(self.cycles)+ " hour cell growth")
            self.ax.set_xlabel(str(self.side*10) +   " micrometers")
            self.ax.set_ylabel(str(self.side*10) +   " micrometers")
        elif mode == 2:
            # Save the current frame
            growth = self.ax.imshow(self.cancer, animated=True)
            immune_coords = np.argwhere(self.immune > 0)
            immune = self.ax.scatter(immune_coords[:,1], immune_coords[:,0], c='blue', s=10)
            self.images.append([growth, immune])
        elif mode == 3:
            # Display the animation
            return animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True)

    # ---------------------------------------------------------------------
    def save_field_to_excel(self, file_name):
        """
        Saves the current state of self.cancer to an excel file.
        
        Parameters:
            file_name (str): name of the excel file
        """

        pd.DataFrame(self.cancer).to_excel(file_name, index=False)

    # ---------------------------------------------------------------------
    def mod_cell(self, x, y, value):
        """
        Modifies cell value. (Create initial state before this!)

        Parameters:
            x, y (int): representing coordinates of the cell
            value (int): the new value at the given position
        """
        
        self.cancer[y][x] = value

    # ---------------------------------------------------------------------
    def get_prolif_potentials(self):
        """
        Returns a dictionary of proliferation potential numbers.
        
        Returns:
            dict: a dictionary of the proliferation potentials
        """
        
        nonzero_field  = np.array(self.cancer)[np.array(self.cancer) > 0]
        unique, counts = np.unique(nonzero_field, return_counts=True)
        prolif_potents = {}
        
        for i in range(1, self.pmax + 2):
            prolif_potents[i] = 0
        for val, count in zip(unique, counts):
            prolif_potents[int(val)] = count
            
        return prolif_potents

    # ---------------------------------------------------------------------
    def get_statistics(self):
        """
        Returns various statistical properties of the model.
        
        Returns:
            dict: a dictionary of the statistical properties
        """
        
        nonzero_field = self.cancer[self.cancer > 0]

        # Statistics
        if nonzero_field.size != 0:
            stats = {
                "Min":        nonzero_field.min(),
                "Max":        nonzero_field.max(),
                "Mean":       nonzero_field.mean(),
                "Std":        nonzero_field.std(),
                "Median":     np.median(nonzero_field),
                "Skew":       skew(nonzero_field.ravel()),
                "Kurtosis":   kurtosis(nonzero_field.ravel()),
                "Final STC":  self.stc_number[self.cycles-1],
                "Final RTC":  self.rtc_number[self.cycles-1],
                "Final WBC":  self.wbc_number[self.cycles-1],
                "Tumor Size": nonzero_field.size,
                "Confluence": nonzero_field.size/self.cancer.size*100,
            }
            
            if self.I > 0:
                stats.update({
                    "Mean I/T"  : sum(self.it_ratio)/len(self.it_ratio),
                    "Mean k/d"  : sum(self.kill_day)/len(self.kill_day)
                })
        
            # Proliferation potentials
            stats.update(self.get_prolif_potentials())
                
            # Cell Numbers
            checkpoints = np.linspace(0, self.cycles - 1, int(self.cycles/10) + 1, dtype=int)
            for idx in checkpoints:
                hour = (idx + 1)
                stats[f"{hour}h_STC"] = self.stc_number[idx]
                stats[f"{hour}h_RTC"] = self.rtc_number[idx]
                stats[f"{hour}h_WBC"] = self.wbc_number[idx]
            
        else: stats = {
            "Tumor Size": 0,
            "Confluence": 0,
            "Status": "Extinct"
            }
        return stats
    
    # ---------------------------------------------------------------------
    def save_statistics(self, file_name):
        """
        Saves various statistical properties of the model to an excel file.
        
        Parameters:
            file_name (str): name of the excel file
        """
        
        stats_dict = self.get_statistics()
        df = pd.DataFrame([stats_dict])
        df.to_excel(file_name, index=False)

    # ---------------------------------------------------------------------
    def measure_runtime(func):
        # Decorator to measure completion time
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result     = func(*args, **kwargs)
            end_time   = time.time()
            runtime    = end_time - start_time
            print("Model completion time (s): " + str(runtime))
            return result
        return wrapper

    # ---------------------------------------------------------------------
    @measure_runtime
    def run_model(self, plot, animate, stats):
        """
        The function that runs a single entire simulation.
        For animation matplotlib backend cannot be inline!

        Parameters:
            plot (bool): set to true to display the plots of the model
            animate (bool): set to true to enable matplotlib animation
            stats (bool): set to true to print statistics of the model
        """

        # Create initial state
        if len(self.cancer) == 0: self.init_state()
        self.find_tumor_cells()
        if len(self.immune) == 0:
            self.immune = np.zeros((self.side, self.side))
        if len(self.mutate) == 0:
            self.mutate = np.zeros((self.side, self.side))
            self.mutmap = np.zeros((self.side, self.side))
        
        self.stc_number = []
        self.rtc_number = []
        self.wbc_number = []
        
        if animate: self.animate(1)

        # Growth loop
        for c in tqdm(range(self.cycles), desc="Running simulation..."):
            self.tumor_action()
            self.immune_response()
            self.find_tumor_cells()
            self.count_tumor_cells()
            if animate: self.animate(2)
        
        # Store the results
        self.store_model()
        
        # Output settings
        if plot: self.plot_run(len(self.runs))
        if animate: self.ani = self.animate(3)
        if stats:
            df = pd.DataFrame(self.stats)
            base_cols = self.separate_columns(df)[0]
            print(df[base_cols])
        
    # ---------------------------------------------------------------------
    @measure_runtime
    def run_multimodel(self, count, init_field, plot, stats):
        """
        Runs the model multiple times and returns a DataFrame of statistics.
    
        Parameters:
            count (int): number of times to run the simulation
            init_field (np.array): custom initial state of field/run
            plot (bool): set to true to display the plots of the model
            stats (bool): set to true to print statistics of the model
    
        Returns:
            pd.DataFrame: collected statistics from each run
        """
        
        stats = []
        
        for i in range(count):
            self.cancer = init_field.copy()
            self.immune = []
            self.mutate = []
            self.mutmap = []
            self.run_model(plot = False, animate = False, stats = False)
            stats.append(self.get_statistics())
        all_stats = pd.DataFrame(stats)
        
        if plot:
            self.plot_averages(all_stats)
        if stats:
            df = pd.DataFrame(self.stats)
            base_cols = self.separate_columns(df)[0]
            print(df[base_cols])
        
        return all_stats

    # ---------------------------------------------------------------------
    def store_model(self):
        """
        Stores the results of the previous model executions.
        """
        
        result = {}
        
        result["immune"] = self.immune
        result["mutate"] = self.mutate
        result["mutmap"] = self.mutmap
        result["cancer"] = self.cancer
        result["stc"]    = self.stc_number
        result["rtc"]    = self.rtc_number
        result["wbc"]    = self.wbc_number
        result["pp"]     = self.get_prolif_potentials().values()
        
        # Stores data for plotting
        self.runs.append(result)     
        
        # Stores data for statistics
        self.stats.append(self.get_statistics())
        
    # ---------------------------------------------------------------------
    def separate_columns(self, data):
        """
        Separates the statistics DataFrame columns into logical groups:
        base stats, STC, RTC, WBC counts, and proliferation potentials.

        Parameters:
            data (pd.DataFrame): Your data in a pandas dataframe format

        Returns:
            tuple of list[str]: A tuple containing 5 lists of column names:
                - base: Columns with general statistical properties
                - stc:  Columns with STC counts at each time point
                - rtc:  Columns with RTC counts at each time point
                - wbc:  Columns with WBC counts at each time point
                - pp:   Columns for proliferation potential values
        """
        
        base = [col for col in data.columns if not str(col).isdigit()
                and "_STC" not in str(col)
                and "_RTC" not in str(col)
                and "_WBC" not in str(col)]
        stc  = sorted([col for col in data.columns if "_STC" in str(col)],
                      key=lambda x: int(str(x).split("h")[0]))
        rtc  = sorted([col for col in data.columns if "_RTC" in str(col)],
                      key=lambda x: int(str(x).split("h")[0]))
        wbc  = sorted([col for col in data.columns if "_WBC" in str(col)],
                      key=lambda x: int(str(x).split("h")[0]))
        pp   = sorted([col for col in data.columns if isinstance(col, int)])
        
        return base, stc, rtc, wbc, pp

    # ---------------------------------------------------------------------
    def plot_run(self, run):
        """
        Creates growth and cell number plots, proliferation potential histograms.
        
        Paramteres:
            run (int): which model execution to plot
            
        Returns:
            matplotlib.figure.Figure: the generated plots of the specific run
        """
        
        # Create the figue and axis
        fig, axs = plt.subplots(2, 2, figsize=(14,14))

        tumor = axs[0, 0].imshow(self.runs[run-1]["cancer"], vmin=0, vmax=self.pmax+1)
        fig.colorbar(tumor, ax=axs[0, 0])
        
        immune_coords = np.argwhere(self.runs[run-1]["immune"] > 0)
        axs[0, 0].scatter(immune_coords[:,1], immune_coords[:,0],
                          c='blue', marker='v', s=10)
        
        axs[0, 1].plot(self.runs[run-1]["stc"], 'C1', label='STC')
        axs[0, 1].plot(self.runs[run-1]["rtc"], 'C2', label='RTC')
        axs[0, 1].plot(self.runs[run-1]["wbc"], 'C3', label='WBC')
        axs[0, 1].legend()
        
        mutmap = axs[1, 0].imshow(self.runs[run-1]["mutmap"],
                 cmap="RdBu_r", vmin=-3, vmax=3, interpolation="bicubic")
        fig.colorbar(mutmap, ax=axs[1, 0])
        
        axs[1, 1].bar(range(1, self.pmax + 2), self.runs[run-1]["pp"], edgecolor='black')

        # Titles/labels of the plots
        titles = [str(self.cycles)+ "h cell growth", "Cell count",
                  "Mutation history", "Final PP values"]
        labs_x = [str(self.side*10) + " um", "Time (h)",
                  str(self.side*10) + " um", "Proliferation potentials"]
        labs_y = [str(self.side*10) + " um", "Cell numbers",
                  str(self.side*10) + " um", "Number of appearance"]

        fig.suptitle("Simulation " + str(run) + " Results", fontsize = 16)
        for i, ax in enumerate(axs.flat):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])

    # ---------------------------------------------------------------------
    def plot_averages(self, data):
        """
        The function that plots the averages of multiple model results.
        Works with the results of the 'run_multimodel' function.
        
        Parameters:
            data (pd.DataFrame): Your data in a pandas dataframe format
            
        Returns:
            matplotlib.figure.Figure: The plots of the averages with SD values
        """
        
        base_cols, stc_cols, rtc_cols, wbc_cols, pp_cols = self.separate_columns(data)
        
        avg_stc = data[stc_cols].mean()
        std_stc = data[stc_cols].std()
        avg_rtc = data[rtc_cols].mean()
        std_rtc = data[rtc_cols].std()
        avg_wbc = data[wbc_cols].mean()
        std_wbc = data[wbc_cols].std()
        avg_pp  = data[pp_cols].mean()
        std_pp  = data[pp_cols].std()
        
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(14, 5))
        timepoints      = np.linspace(0, self.cycles - 1, int(self.cycles/10) + 1)
        
        ax1.plot(timepoints, avg_stc, label='STC', color='C1')
        ax1.fill_between(timepoints, avg_stc - std_stc, avg_stc + std_stc,
                         color='C1', alpha=0.3)
        ax1.plot(timepoints, avg_rtc, label='RTC', color='C2')
        ax1.fill_between(timepoints, avg_rtc - std_rtc, avg_rtc + std_rtc,
                         color='C2', alpha=0.3)
        ax1.plot(timepoints, avg_wbc, label='WBC', color='C3')
        ax1.fill_between(timepoints, avg_wbc - std_wbc, avg_wbc + std_wbc,
                         color='C3', alpha=0.3)
        
        ax1.set_title("Average Tumor Cell Count")
        ax1.set_xlabel("Model Time (hours)")
        ax1.set_ylabel("Number of Cells")
        ax1.legend()

        ax2.bar(pp_cols, avg_pp, yerr=std_pp, capsize=5, edgecolor='black')
        ax2.set_title("Average Proliferation Potential Distribution")
        ax2.set_xlabel("Proliferation Potential")
        ax2.set_ylabel("Average Count")
        
        fig.suptitle("Averages of " + str(len(self.stats)) + " Models", fontsize = 16)
        plt.tight_layout()


class TDashboard:
    """
    Class for a Streamlit dashboard providing a GUI for the model.
    
    Parameters:
        model (TModel): The created model you want a dashboard for
    """
    
    def __init__(self, model):
        self.model = model

    # ---------------------------------------------------------------------
    def run_dashboard(self):
        """
        The function that creates the entire streamlit dashboard for the model.
        """
        
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='text-align: center;'>TCAMpy</h1>", unsafe_allow_html=True)
        self.screen_width = st_javascript("window.innerWidth", key="screen_width")

        tab1, tab2 = st.tabs(["SIMULATION", "MACHINE LEARNING"])
        with tab1:
            self.columns = [4, 1, 12]
            self.col1, _, self.col3 = st.columns(self.columns)

            with self.col1:
                self._initialize()
                self._modify_cell()
                self._execute_model()
            with self.col3:
                self._visualize_run("Last Simulation", len(self.model.runs))
                self._show_statistics()
                self._reset_save_stats()
        
        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                self._simdata_generator()
            with col2:
                self._train_and_predict()

    # ---------------------------------------------------------------------
    def print_title(self, title):
        """
        The function that prints text as a title on the dashboard.
        
        Parameters:
            title (string): The text to print
        """

        st.markdown(
            f"<h2 style='text-align: center;'>{title}</h2>",
            unsafe_allow_html=True
        )
    
    # ---------------------------------------------------------------------
    def get_plot_height(self, col, scaler):
        """
        The function that calculates the height of plots
        based on screen width, column width and a scaler.
        
        Parameters:
            col (int): main column number
            scalar (float): scaler for column width
        """
        
        screen_width = st.session_state.get("screen_width")
        col_width_px = screen_width * (self.columns[col-1] / sum(self.columns))
        return int(col_width_px * scaler)

    # ---------------------------------------------------------------------
    def _initialize(self):
        """
        The function that sets the parameters and initializes the model.
        """

        self.print_title("Model Parameters")

        self.model.cycles = st.slider("Model Duration (hours)", 50, 5000, value=self.model.cycles)
        self.model.side   = st.slider("Field Side Length (10um)", 10, 200, value=self.model.side)
        self.model.pmax   = st.slider("Max Proliferation Potential", 1, 20, value=self.model.pmax)
        self.model.PA     = st.slider("Apoptosis Chance (RTC) (%)", 0, 100, value=self.model.PA)
        self.model.CCT    = st.slider("Cell Cycle Time (hours)", 1, 48, value=self.model.CCT)
        self.model.Dt     = st.slider("Time Step (days)", 0.01, 1.0, value=self.model.Dt, step=0.01)
        self.model.PS     = st.slider("STC-STC Division Chance (%)", 0, 100, value=self.model.PS)
        self.model.mu     = st.slider("Migration Capacity", 0, 10, value=self.model.mu)
        self.model.I      = st.slider("Immune Strength", 0, 10, value=self.model.I)
        self.model.M      = st.slider("Mutation Chance", 0, 50, value=self.model.M)

        self.model.PP = int(self.model.CCT * self.model.Dt / 24 * 100)
        self.model.PM = 100 * self.model.mu / 24

        init_config = (
            self.model.side, self.model.cycles, self.model.pmax,
            self.model.PA, self.model.CCT, self.model.Dt, self.model.PS,
            self.model.mu, self.model.I, self.model.M
        )
        config_hash = hashlib.md5(str(init_config).encode()).hexdigest()

        # Storing data for model plotting
        if "model_runs" in st.session_state:
            self.model.runs = st.session_state.model_runs
            
        # Storing data for model statistics
        if "model_stats" in st.session_state:
            self.model.stats = st.session_state.model_stats

        if (
            "initialized" not in st.session_state
            or "init_config_hash" not in st.session_state
            or st.session_state.init_config_hash != config_hash
        ):
            self.model.init_state()
            st.session_state.cancer  = self.model.cancer.copy()
            st.session_state.immune = self.model.immune.copy()
            st.session_state.mutate = self.model.mutate.copy()
            st.session_state.mutmap = self.model.mutmap.copy()
            st.session_state.initialized = True
            st.session_state.init_config_hash = config_hash

    # ---------------------------------------------------------------------
    def _modify_cell(self):
        """
        The function for initial state modification logic.
        """
        
        self.print_title("Initial State")

        x_coord = st.number_input("X Coordinate", 0, self.model.side - 1, value=self.model.side // 2)
        y_coord = st.number_input("Y Coordinate", 0, self.model.side - 1, value=self.model.side // 2)
        cell_value = st.number_input("Cell Value", 0, self.model.pmax + 1, value=self.model.pmax + 1)
        plots_height = self.get_plot_height(1, 0.9)

        if st.button("Modify Cell"):
            self.model.cancer = st.session_state.cancer.copy()
            self.model.mod_cell(x_coord, y_coord, cell_value)
            st.session_state.cancer = self.model.cancer.copy()
            st.success(f"Cell modified at ({x_coord}, {y_coord}) to {cell_value}")

        cancer  = st.session_state.cancer
        heatmap = self._create_heatmap(
            plots_height, "Initial state", "viridis",
            "PP", 0, self.model.pmax+1, cancer
            )
        
        st.altair_chart(heatmap, use_container_width=True)

    # ---------------------------------------------------------------------
    def _execute_model(self):
        """
        The function for model running logic.
        """
        
        self.print_title("Execution")
        
        rep = st.number_input("How many simulations?", 1)
        
        if st.button("Run Model"):
            with st.spinner("Running simulations..."):
                for i in range(rep):
                    self.model.cancer  = st.session_state.cancer.copy()
                    self.model.immune = st.session_state.immune.copy()
                    self.model.mutate = st.session_state.mutate.copy()
                    self.model.mutmap = st.session_state.mutmap.copy()
                    self.model.run_model(plot = False, animate=False, stats=False)
    
                st.session_state.model_runs = self.model.runs
                st.session_state.model_stats = self.model.stats

    # ---------------------------------------------------------------------
    def _visualize_run(self, title, run):
        """
        The function for the result visualization logic.
    
        Parameters:
            title (string): title of the visualization
            run (int): which model execution to plot
        """
    
        if "model_runs" not in st.session_state:
            st.warning("Simulation results will appear here...")
            return
        self.print_title(title)
    
        # --- Get latest run ---
        latest = self.model.runs[run - 1]
        immune = latest["immune"]
        mutmap = latest["mutmap"]
        cancer = latest["cancer"]
        stc    = latest["stc"]
        rtc    = latest["rtc"]
        wbc    = latest["wbc"]
        pp     = latest["pp"]
    
        # --- Create charts ---
        plots_height = self.get_plot_height(3, 0.4)
        
        tumor_heatmap = self._create_heatmap(
            plots_height, "Tumor growth", "viridis",
            "PP", 0, self.model.pmax+1, cancer, immune
        )
        mutation_map = self._create_heatmap(
            plots_height, "Mutation history", "redblue",
            "M", -3, 3, mutmap
        )
        
        bar_chart  = self._create_bar_chart(plots_height, list(pp))
        line_chart = self._create_line_chart(plots_height, stc, rtc, wbc)

        # --- Layout rules ---
        col1, col2 = st.columns([4, 5])
        with col1:
            st.altair_chart(tumor_heatmap, use_container_width=True)
            st.altair_chart(mutation_map, use_container_width=True)
        with col2:
            st.altair_chart(bar_chart, use_container_width=True)
            st.altair_chart(line_chart, use_container_width=True)

    # ---------------------------------------------------------------------
    def _create_heatmap(
            self, h, title, cmap, ctitle,
            vmin, vmax, heatmap, scatter=None
        ):
        """
        Creates an Altair heatmap with a scatter plot overlaid.
        Used for tumor field with immune cells, and mutations.
    
        Parameters:
            h (int): the height of the plot
            title (string): title of the plot
            cmap (stirng): colormap for the heatmap
            vmin, vmax (int): domain for the colormap
            heatmap (2D array-like): array for the heatmap
            scatter (2D array-like): array for scatter plot
    
        Returns:
            Altair.Chart: heatmap with scatter overlay
        """
        
        # --- Heatmap data ---
        heat_df = pd.DataFrame([
            {"x": x, "y": y, "value": heatmap[y, x]}
            for y in range(heatmap.shape[0])
            for x in range(heatmap.shape[1])
        ])
    
        heat_chart = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X("x:O", title="X"),
            y=alt.Y("y:O", sort="descending", title="Y"),
            color=alt.Color("value:Q", title=ctitle,
                  scale=alt.Scale(scheme=cmap, domain=[vmin, vmax]))
        ).properties(
            title = title,
            width='container',
            height=h
        )
        
        # --- Scatter plot data ---
        if scatter is not None:
            scatter_coords = np.argwhere(scatter > 0)
            scatter_df = pd.DataFrame(scatter_coords, columns=["y", "x"])
        
            scatter = alt.Chart(scatter_df).mark_point(
                color="blue", size=h/20, filled=True, shape="circle"
            ).encode(
                x=alt.X("x:O"),
                y=alt.Y("y:O", sort="descending")
            )
        
            # --- Combine layers ---
            heat_chart = (heat_chart + scatter).properties(
                title=title,
                width='container',
                height=h
            )
    
        return heat_chart
    
    # ---------------------------------------------------------------------
    def _create_line_chart(
            self, h, stc, rtc, wbc, stc_l=None, stc_u=None,
            rtc_l=None, rtc_u=None, wbc_l=None, wbc_u=None
        ):
        """
        The function that creates an Altair line chart of the cell numbers.
        
        Parameters:
            h (int): the height of the plot
            stc, rtc, wbc (list): a list of the cell and immune numbers (mean or raw)
            stc_l, rtc_l, wbc_l (list of float, optional): Lower bounds (e.g., mean - SD) for cell counts.
            stc_u, rtc_u, wbc_u (list of float, optional): Upper bounds (e.g., mean + SD) for cell counts.
            
        Returns:
            Altair.Chart: represents the line chart of the cell numbers
        """

        timepoints = list(range(len(stc)))
        df = pd.DataFrame({
            "Hour": timepoints * 3,
            "Cell Type": ["STC"] * len(stc) + ["RTC"] * len(rtc) + ["WBC"] * len(wbc),
            "Mean": stc + rtc + wbc
        })
    
        if stc_l and rtc_l and wbc_l:
            df["Lower"] = stc_l + rtc_l + wbc_l
            df["Upper"] = stc_u + rtc_u + wbc_u
    
            area = alt.Chart(df).mark_area(opacity=0.3).encode(
                x=alt.X("Hour:Q", title="Time (hours)"),
                y=alt.Y("Lower:Q", title="Mean"),
                y2="Upper:Q",
                color="Cell Type:N"
            )
        else:
            area = None
    
        line = alt.Chart(df).mark_line().encode(
            x="Hour:Q",
            y=alt.Y("Mean:Q", title="Mean"),
            color="Cell Type:N"
        )

        chart = (area + line) if area else line
        return chart.properties(title="Cell Counts Over Time", height=h)
    
    # ---------------------------------------------------------------------
    def _create_bar_chart(self, h, pp, std=None):
        """
        Creates an Altair bar chart for proliferation potential distribution.
    
        Parameters:
            h (int): the height of the plot
            pp (list of float or int): Mean or raw counts of cells per proliferation potential class
            std (list of float, optional): Standard deviation for each class
    
        Returns:
            alt.Chart: An Altair chart representing the distribution of proliferation potentials
        """
        
        pp_df = pd.DataFrame({
            "Proliferation Potential": list(range(1, len(pp) + 1)),
            "Mean": pp
        })
        chart = alt.Chart(pp_df).mark_bar().encode(
            x="Proliferation Potential:O",
            y="Mean:Q"
        )
    
        if std is not None:
            pp_df["Std"] = std
            error = alt.Chart(pp_df).mark_errorbar(extent="stdev").encode(
                x="Proliferation Potential:O",
                y="Mean:Q",
                yError="Std:Q"
            )
            chart = chart + error
    
        return chart.properties(title="Proliferation Potential Distribution", height=h)

    # ---------------------------------------------------------------------
    def _show_statistics(self):
        """
        The function for the statistics printing logic.
        """
        
        if not self.model.stats: return
        
        self.print_title("All Simulations")
        plots_height = self.get_plot_height(3, 0.4)
        df = pd.DataFrame(self.model.stats)
        base_cols, stc_cols, rtc_cols, wbc_cols, pp_cols = self.model.separate_columns(df)
        df.index = df.index + 1

        # Display Statistics
        mean_row = df[base_cols].mean(numeric_only=True)
        std_row  = df[base_cols].std(numeric_only=True)
        mean_row.name = "Mean"
        std_row.name  = "Std"
        full_stats = pd.concat([df[base_cols], mean_row.to_frame().T, std_row.to_frame().T])
        st.dataframe(full_stats)

        # Create avg charts
        stc_means = df[stc_cols].mean()
        stc_stds  = df[stc_cols].std()
        rtc_means = df[rtc_cols].mean()
        rtc_stds  = df[rtc_cols].std()
        wbc_means = df[wbc_cols].mean()
        wbc_stds  = df[wbc_cols].std()
        pp_means  = df[pp_cols].mean()
        pp_stds   = df[pp_cols].std()
        
        line_chart = self._create_line_chart(plots_height,
            list(stc_means.values), list(rtc_means.values), list(wbc_means.values),
            list((stc_means - stc_stds).values), list((stc_means + stc_stds).values),
            list((rtc_means - rtc_stds).values), list((rtc_means + rtc_stds).values),
            list((wbc_means - wbc_stds).values), list((wbc_means + wbc_stds).values)
        )
        
        bar_chart = self._create_bar_chart(plots_height, list(pp_means.values), list(pp_stds.values))
        
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(line_chart, use_container_width=True)
        with col2:
            st.altair_chart(bar_chart, use_container_width=True)

    # ---------------------------------------------------------------------
    def _reset_save_stats(self):
        """
        The function for the reset/download statistics logic.
        """
        
        if "model_stats" in st.session_state:
            self.print_title("Simulation Options")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            selected_run = None
            visualize = False
            
            with col1:
                if st.button("Reset Model Executions and Data", use_container_width=True):
                    del st.session_state.model_stats
                    self.model.stats.clear()
                    del st.session_state.model_runs
                    self.model.runs.clear()
                    
                    st.success("Executions have been reset.")
            with col2:
                buffer = io.BytesIO()
                pd.DataFrame(self.model.stats).to_excel(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="Download Statistics (xlsx)",
                    data=buffer,
                    file_name="simulation_statistics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col3:
                run_select = st.selectbox(
                    "", list(range(1, len(self.model.runs) + 1)),
                    placeholder="Select simulation",
                    label_visibility="collapsed",
                    index=None
                    )
                selected_run = run_select
            with col4:
                if st.button("Visualize Selected Simulation", use_container_width=True):
                    if selected_run: visualize = True
                    else: st.warning('Please select a simulation!')
            if visualize:
                self._visualize_run("Selected Simulation", selected_run)

    # ---------------------------------------------------------------------
    def _simdata_generator(self):
        """
        The Machine Learning tab for dataset generation and download.
        Uses the TML class to generate simulation data.
        """

        self.print_title("Simulation Data Generator")

        # Initialize TML
        tml = TML(self.model)

        st.write("Select randomization ranges for each parameter:")

        # Build parameter range inputs dynamically
        param_ranges = {}
        for param, default_val in tml.default_params.items():
            col1, col2 = st.columns(2)
            with col1:
                low = st.number_input(
                    f"{param} (min)", 
                    value=float(default_val) * 0.8, 
                    key=f"{param}_low"
                )
            with col2:
                high = st.number_input(
                    f"{param} (max)", 
                    value=float(default_val) * 1.5, 
                    key=f"{param}_high"
                )
            param_ranges[param] = (low, high)

        n = st.number_input("Number of simulations", 5, 500, 50, step=5)

        # Run simulation button
        if st.button("Generate Dataset", use_container_width=True):
            with st.spinner("Running simulations..."):
                df = tml.generate_dataset(n=n, random_params=param_ranges)
            st.success(f"Dataset generated successfully ({len(df)} rows).")

            # Allow CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Dataset (.csv)",
                data=csv,
                file_name="tumor_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------------------------------------------------------------------
    def _train_and_predict(self):
        """
        Streamlit UI for model training and prediction using the TML class.
        """
        
        self.print_title("Model Trainer and Predictor")

        tml = TML(self.model)
    
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset preview:")
            st.dataframe(df.head())
    
            # Choose target column
            target = st.selectbox("Select target attribute", df.columns, index=len(df.columns) - 1)
    
            # Three sliders for model parameters
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
            random_state = st.slider("Random Seed", 0, 100, 42, step=1)
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 200, step=50)
    
            if st.button("Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    model, metrics = tml.train_predictor(
                        file=df,
                        target=target,
                        test_size=test_size,
                        random_state=random_state,
                        n_estimators=n_estimators
                    )
    
                st.success("Model trained successfully!")
                st.write(f"**R^2:** {metrics['R2']:.3f}")
                st.write(f"**MAE:** {metrics['MAE']:.3f}")
    
                # Store trained model for later prediction
                st.session_state["trained_tml"] = tml
                st.session_state["target"] = target
        else:
            st.info("Please upload a dataset to train a model.")

        if "trained_tml" in st.session_state:
            trained_tml = st.session_state["trained_tml"]
            target = st.session_state.get("target", "Target")
            feature_cols = trained_tml.feature_columns
    
            # Numeric inputs for each feature
            new_params = []
            self.print_title("Predict for new instance")
            for col in feature_cols:
                val = st.number_input(f"{col}", value=1.0, key=f"pred_{col}")
                new_params.append(val)
    
            # Target selector (for user clarity, though single-target regression)
            st.markdown("#### Select Target to Predict:")
            st.text(f"Predicting: {target}")
    
            if st.button("🔮 Predict New", use_container_width=True):
                try:
                    prediction = trained_tml.predict_new(new_params)
                    st.success(f"Predicted {target}: **{prediction:.3f}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Train a model first to enable prediction.")


class TML:
    """
    Class for handling Machine Learning tasks related to the tumor model.
    Allows dataset generation, parameter exploration, and result export.
    Allows predicting the size/confluence for a new set of parameters.
    
    Parameters:
        model (TModel): a created instance of the TModel class.
    """

    def __init__(self, model):
        self.model = model

        self.default_params = {
            "cycles": self.model.cycles,
            "side":   self.model.side,
            "pmax":   self.model.pmax,
            "PA":     self.model.PA,
            "CCT":    self.model.CCT,
            "Dt":     self.model.Dt,
            "PS":     self.model.PS,
            "mu":     self.model.mu,
            "I":      self.model.I,
            "M":      self.model.M,
        }

    # ---------------------------------------------------------------------
    def generate_dataset(
            self, n=50, random_params=None,
            output_file="tumor_dataset.csv"
        ):
        """
        Generate a dataset of tumor simulations by randomizing given parameters.

        Parameters:
            n_sims (int): Number of simulations to run.
            randomize_params (dict): Parameters to randomize, e.g.
                {
                    "PA": (1, 20), "PS": (10, 40))
                }
            output_file (str): CSV filename to save dataset.

        Returns:
            pd.DataFrame: Combined DataFrame with all simulation results.
        """

        stats = []

        # Randomize chosen parameters
        for i in tqdm(range(n), desc="Generating simulations"):
            params = self.default_params.copy()
            for key, (low, high) in random_params.items():
                if isinstance(params[key], int):
                    params[key] = random.randint(int(low), int(high))
                else:
                    params[key] = random.uniform(float(low), float(high))

            # Run simulation
            model = TModel(**params)
            model.run_model(plot = False, animate = False, stats = False)

            run_stats = {}
            for k, v in params.items():
                run_stats[k] = v
            run_stats["Tumor size"] = np.count_nonzero(model.cancer)
            run_stats["Confluence"] = np.count_nonzero(model.cancer)/model.cancer.size*100
            stats.append(run_stats)

        if stats:
            df = pd.DataFrame(stats)
            df.to_csv(output_file, index=False)
            print(f"Dataset saved to {output_file} ({len(df)} runs)")
            return df
            
    # ---------------------------------------------------------------------
    def train_predictor(
            self, file, target, test_size=0.2, 
            random_state=42, n_estimators=200
        ):
        """
        Trains a regression model to predict final tumor size based on simulation parameters.
    
        Parameters:
            file (str): CSV file containing the dataset
            target (str): Column name of the target attribute
            test_size (float): Fraction of dataset to use for testing
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of trees in the random forest
    
        Returns:
            model (RandomForestRegressor): Trained model
            metrics (dict): R^2 and MAE metrics on test set
        """

        if isinstance(file, pd.DataFrame):
            df = file
        else:
            df = pd.read_csv(file)
        x  = df[df.columns[0:10]]
        y  = df[target]
    
        self.feature_columns = x.columns.tolist()

        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(x_train, y_train)
    
        # Predict & evaluate
        y_pred = model.predict(x_test)
        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred)
        }
        self.trained_model = model
    
        print(f"Model trained on {len(x_train)} samples, tested on {len(x_test)}")
        print(f"R^2: {metrics['R2']:.3f}, MAE: {metrics['MAE']:.3f}")
    
        # Feature importance summary
        importance = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
        print("\n Top influencing parameters:")
        print(importance.head())
    
        return model, metrics
    
    # ---------------------------------------------------------------------
    def predict_new(self, params):
        """
        Predicts an attribute value for a set of
        parameters using a previously trained model.

        Parameters:
            params (list): List of parameters, e.g.
                [500, 50, 10, 1, 24, 1/24, 15, 4, 4, 10]

        Returns:
            float: Predicted tumor size or confluence
        """
        if self.trained_model is None:
            raise RuntimeError(
                "No trained model found. Train one with train_predictor() first."
                )
        if self.feature_columns is None:
            raise RuntimeError(
                "Feature column list not found. Did you train the model?"
                )

        if len(params) != len(self.feature_columns):
            raise ValueError(
                f"Parameter list must have {len(self.feature_columns)} values "
                f"(got {len(params)}). Expected order: {self.feature_columns}"
            )
        df = pd.DataFrame([params], columns=self.feature_columns)

        # Ensure all expected features are present (fill missing ones with 0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        return self.trained_model.predict(df)[0]