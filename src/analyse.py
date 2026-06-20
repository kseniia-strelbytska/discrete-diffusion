import os
import sys
import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class CSVAnalyzerApp:
    def __init__(self, root, initial_filepath=None):
        self.root = root
        self.root.title("Research Data Analyzer")
        self.root.geometry("1100x700")
        
        self.data = []
        self.headers = []
        self.current_filtered_data = [] # Tracks the currently filtered subset
        self.filter_vars = {}           # Stores string variables for every dynamic filter
        
        # --- Top Frame: Action Buttons ---
        action_frame = tk.Frame(root)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Button(action_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Reset Filters", command=self.reset_filters).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Identify Best Models (In Current View)", 
                  command=self.identify_best, bg="#add8e6").pack(side=tk.RIGHT, padx=5)
        
        # --- Middle Frame: Dynamic Filters ---
        # We use a frame to hold a grid of automatically generated comboboxes
        self.filter_frame = tk.Frame(root)
        self.filter_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # --- Bottom Frame: Data Table ---
        tree_frame = tk.Frame(root)
        tree_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        y_scroll = tk.Scrollbar(tree_frame)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)

        # Automatically load file if passed via command line
        if initial_filepath:
            self.load_file_from_path(initial_filepath)

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.load_file_from_path(filepath)

    def load_file_from_path(self, filepath):
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"The file path provided does not exist:\n{filepath}")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.headers = next(reader)
                self.data = [row for row in reader if row]
                self.current_filtered_data = self.data.copy()
                
            self.setup_treeview()
            self.setup_dynamic_filters()
            self.populate_tree(self.data)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    def setup_treeview(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = self.headers
        self.tree["show"] = "headings"
        
        for col in self.headers:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))
            self.tree.column(col, width=90, anchor=tk.CENTER)

    def setup_dynamic_filters(self):
        # Clear existing filters in case a new file is loaded
        for widget in self.filter_frame.winfo_children():
            widget.destroy()
            
        self.filter_vars.clear()
        
        if not self.headers: 
            return

        # Arrange filters in a grid (e.g., 5 columns wide to prevent UI stretching)
        max_columns = 5
        row, col = 0, 0
        
        for i, header in enumerate(self.headers):
            # Extract unique, sorted values for this specific column
            unique_values = sorted(list(set(row[i] for row in self.data if row[i])))
            
            # Filter Label
            tk.Label(self.filter_frame, text=f"{header}:").grid(row=row, column=col*2, sticky=tk.E, padx=(10, 2), pady=2)
            
            # Combobox Variable and Widget
            var = tk.StringVar()
            combo = ttk.Combobox(self.filter_frame, textvariable=var, state="readonly", width=10)
            combo["values"] = ["All"] + unique_values
            combo.grid(row=row, column=col*2+1, sticky=tk.W, padx=(0, 10), pady=2)
            
            var.set("All")
            combo.bind("<<ComboboxSelected>>", self.apply_filters)
            
            self.filter_vars[header] = var
            
            # Grid layout logic wrapping
            col += 1
            if col >= max_columns:
                col = 0
                row += 1

    def apply_filters(self, event=None):
        if not self.data: return
        
        filtered = []
        
        for row in self.data:
            match = True
            for i, header in enumerate(self.headers):
                selected_val = self.filter_vars[header].get()
                
                # If the dropdown isn't 'All' and the row value doesn't match, discard the row
                if selected_val != "All" and row[i] != selected_val:
                    match = False
                    break 
                    
            if match:
                filtered.append(row)
                
        self.current_filtered_data = filtered
        self.populate_tree(self.current_filtered_data)

    def reset_filters(self):
        for var in self.filter_vars.values():
            var.set("All")
        self.current_filtered_data = self.data.copy()
        self.populate_tree(self.data)

    def identify_best(self):
        # Operates ONLY on the currently filtered data
        if not self.current_filtered_data: 
            messagebox.showinfo("Info", "No data to analyze in the current filter.")
            return
            
        try:
            acc_idx = self.headers.index("mean_both_rules_acc")
            time_idx = self.headers.index("elapsed_s")
            
            # Sort primary by Accuracy (Descending) and secondary by Time (Ascending)
            # Safe float parsing handles empty string blanks like those in 'sigma' and 'eb_gamma'
            sorted_data = sorted(
                self.current_filtered_data, 
                key=lambda x: (
                    float(x[acc_idx]) if x[acc_idx].strip() else -1.0, 
                    -float(x[time_idx]) if x[time_idx].strip() else -float('inf')
                ),
                reverse=True
            )
            
            # Update the current filter view to the sorted view
            self.current_filtered_data = sorted_data
            self.populate_tree(sorted_data)
            
            # Highlight the top 3 performers in the subset
            for i, item in enumerate(self.tree.get_children()[:3]):
                self.tree.item(item, tags=('best',))
            self.tree.tag_configure('best', background='#d4edda')
            
        except ValueError:
            messagebox.showwarning("Warning", "Required columns (mean_both_rules_acc, elapsed_s) not found.")

    def populate_tree(self, data_subset):
        self.tree.delete(*self.tree.get_children())
        for row in data_subset:
            self.tree.insert("", tk.END, values=row)

    def sort_column(self, col, reverse):
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0]) if t[0].strip() else -float('inf'), reverse=reverse)
        except ValueError:
            items.sort(key=lambda t: t[0].lower(), reverse=reverse)
            
        for index, (val, k) in enumerate(items):
            self.tree.move(k, "", index)
            
        self.tree.heading(col, command=lambda: self.sort_column(col, not reverse))


if __name__ == "__main__":
    root = tk.Tk()
    target_file = None
    
    if len(sys.argv) > 1:
        input_string = sys.argv[1]
        
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        results_dir = project_root / "results"
        
        path_obj = Path(input_string)
        
        if path_obj.exists():
            target_file = str(path_obj.resolve())
        else:
            print(f"Searching for '{input_string}' in {results_dir}...")
            matches = list(results_dir.rglob(f"*{path_obj.name}"))
            
            if matches:
                target_file = str(matches[0])
                print(f"Found match: {target_file}")
            else:
                print(f"Warning: Could not find anything matching '{input_string}' in {results_dir}")

    app = CSVAnalyzerApp(root, initial_filepath=target_file)
    root.mainloop()