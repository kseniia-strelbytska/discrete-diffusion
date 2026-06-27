""" 
Tkinter GUI for displaying CSV results from the research experiments.
Filter by columns, identify the best model from filtered.
Load the csv directly from the command line or select it in GIU.

"""

import os
import sys
import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class ScrollableFrame(ttk.Frame):
    """A helper class to create a scrollable frame for our massive list of filters."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class CSVAnalyzerApp:
    def __init__(self, root, initial_filepath=None):
        self.root = root
        self.root.title("Research Data Analyzer Pro")
        self.root.geometry("1300x750")
        
        self.data = []
        self.headers = []
        self.current_filtered_data = [] 
        self.filter_vars = {}           
        self.column_visibility_vars = {} # Tracks which columns are visible
        
        # --- Top Frame: Action Buttons ---
        action_frame = tk.Frame(root, padx=10, pady=10)
        action_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(action_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Select Columns", command=self.open_column_selector).pack(side=tk.LEFT, padx=5)
        tk.Button(action_frame, text="Reset Filters", command=self.reset_filters).pack(side=tk.LEFT, padx=5)
        
        tk.Button(action_frame, text="Identify Best Models", 
                  command=self.identify_best, bg="#add8e6", font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=5)
        
        # --- Main Body (PanedWindow for resizable sidebar) ---
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Left Panel: Scrollable Filters
        self.sidebar = ScrollableFrame(self.paned_window)
        self.paned_window.add(self.sidebar, weight=0) # weight=0 prevents sidebar from expanding too much
        
        # Right Panel: Data Table
        tree_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(tree_frame, weight=1)
        
        y_scroll = tk.Scrollbar(tree_frame)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)

        if initial_filepath:
            self.load_file_from_path(initial_filepath)

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.load_file_from_path(filepath)

    def load_file_from_path(self, filepath):
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"Path does not exist:\n{filepath}")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.headers = next(reader)
                self.data = [row for row in reader if row]
                self.current_filtered_data = self.data.copy()
            
            # Default: Show all columns initially
            self.column_visibility_vars = {h: tk.BooleanVar(value=True) for h in self.headers}
                
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
            self.tree.column(col, width=120, anchor=tk.CENTER)
            
        self.update_display_columns()

    def update_display_columns(self):
        """Updates the treeview to only show columns checked by the user."""
        visible_cols = [h for h in self.headers if self.column_visibility_vars[h].get()]
        
        # Tkinter requires an empty string tuple to hide all, but we avoid hiding everything
        if not visible_cols:
            messagebox.showwarning("Warning", "You must select at least one column.")
            return
            
        self.tree["displaycolumns"] = visible_cols

    def open_column_selector(self):
        """Opens a popup window to toggle column visibility."""
        if not self.headers: return
        
        top = tk.Toplevel(self.root)
        top.title("Select Columns to Display")
        top.geometry("350x500")
        
        tk.Label(top, text="Select metrics to analyze:", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Scrollable area for checkboxes
        scroll_frame = ScrollableFrame(top)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        for header in self.headers:
            chk = tk.Checkbutton(
                scroll_frame.scrollable_frame, 
                text=header, 
                variable=self.column_visibility_vars[header]
            )
            chk.pack(anchor=tk.W, pady=1)
            
        tk.Button(top, text="Apply Changes", command=lambda: [self.update_display_columns(), top.destroy()], bg="#add8e6").pack(pady=10)

    def setup_dynamic_filters(self):
        for widget in self.sidebar.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.filter_vars.clear()
        if not self.headers: return

        tk.Label(self.sidebar.scrollable_frame, text="Filters", font=("Arial", 12, "bold")).pack(pady=(5, 10))

        for i, header in enumerate(self.headers):
            unique_values = sorted(list(set(row[i] for row in self.data if row[i])))
            
            # Group label and dropdown tightly together
            frame = tk.Frame(self.sidebar.scrollable_frame)
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(frame, text=header).pack(anchor=tk.W)
            
            var = tk.StringVar()
            combo = ttk.Combobox(frame, textvariable=var, state="readonly", width=25)
            combo["values"] = ["All"] + unique_values
            combo.pack(fill=tk.X)
            
            var.set("All")
            combo.bind("<<ComboboxSelected>>", self.apply_filters)
            self.filter_vars[header] = var

    def apply_filters(self, event=None):
        if not self.data: return
        
        filtered = []
        for row in self.data:
            match = True
            for i, header in enumerate(self.headers):
                selected_val = self.filter_vars[header].get()
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
        if not self.current_filtered_data: 
            return
            
        try:
            # Fallback logic in case you use an older dataset structure
            if "mean_both_rules" in self.headers:
                acc_idx = self.headers.index("mean_both_rules")
            else:
                acc_idx = self.headers.index("mean_both_rules_acc")
                
            time_idx = self.headers.index("elapsed_s")
            
            sorted_data = sorted(
                self.current_filtered_data, 
                key=lambda x: (
                    float(x[acc_idx]) if x[acc_idx].strip() else -1.0, 
                    -float(x[time_idx]) if x[time_idx].strip() else -float('inf')
                ),
                reverse=True
            )
            
            self.current_filtered_data = sorted_data
            self.populate_tree(sorted_data)
            
            for i, item in enumerate(self.tree.get_children()[:3]):
                self.tree.item(item, tags=('best',))
            self.tree.tag_configure('best', background='#d4edda')
            
        except ValueError:
            messagebox.showwarning("Warning", "Required columns (mean_both_rules, elapsed_s) not found.")

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
            else:
                print(f"Warning: Could not find '{input_string}'")

    app = CSVAnalyzerApp(root, initial_filepath=target_file)
    root.mainloop()