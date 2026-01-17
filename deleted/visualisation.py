import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
from pathlib import Path

# Token definitions
TOKEN_MAP = {
    2: {'name': 'EOS', 'color': '#fee2e2', 'border': '#fca5a5', 'text': '#991b1b', 'desc': 'End of Sequence'},
    3: {'name': 'SOS', 'color': '#dcfce7', 'border': '#86efac', 'text': '#166534', 'desc': 'Start of Sequence'},
    4: {'name': 'PAD', 'color': '#f3f4f6', 'border': '#d1d5db', 'text': '#4b5563', 'desc': 'Padding Token'},
    5: {'name': 'MASK', 'color': '#fef3c7', 'border': '#fde047', 'text': '#854d0e', 'desc': 'Masked Token (Noise)'},
}

class DenoisingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Discrete Diffusion Denoising Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#eff6ff')
        
        self.steps = []
        self.current_step = 0
        self.is_playing = False
        self.speed = 800  # milliseconds
        self.play_job = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#eff6ff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Discrete Diffusion Denoising Process", 
                               font=('Arial', 24, 'bold'), bg='#eff6ff', fg='#1e3a8a')
        title_label.pack(pady=(0, 10))
        
        # Goal description
        goal_frame = tk.Frame(main_frame, bg='#dbeafe', relief=tk.SOLID, borderwidth=2)
        goal_frame.pack(fill=tk.X, pady=(0, 15))
        
        goal_title = tk.Label(goal_frame, text="Model Goal:", 
                             font=('Arial', 12, 'bold'), bg='#dbeafe', fg='#1e40af')
        goal_title.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        goal_text = tk.Label(goal_frame, 
                            text="Unmask a string in such a way so that the number of 0s and 1s match and all 0s go before all 1s.",
                            font=('Arial', 11), bg='#dbeafe', fg='#1e40af', wraplength=1200, justify=tk.LEFT)
        goal_text.pack(anchor=tk.W, padx=10, pady=(0, 10))
        
        # Load file button
        button_frame = tk.Frame(main_frame, bg='#eff6ff')
        button_frame.pack(pady=(0, 10))
        
        load_btn = tk.Button(button_frame, text="Load File", command=self.load_file,
                            font=('Arial', 12), bg='#4f46e5', fg='white', 
                            padx=20, pady=10, relief=tk.FLAT, cursor='hand2')
        load_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(button_frame, text="▶ Play", command=self.toggle_play,
                                 font=('Arial', 12), bg='#4f46e5', fg='white',
                                 padx=20, pady=10, relief=tk.FLAT, cursor='hand2', state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = tk.Button(button_frame, text="↻ Reset", command=self.reset,
                             font=('Arial', 12), bg='#6b7280', fg='white',
                             padx=20, pady=10, relief=tk.FLAT, cursor='hand2')
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = tk.Frame(button_frame, bg='#eff6ff')
        speed_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(speed_frame, text="Speed:", font=('Arial', 11), bg='#eff6ff').pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.IntVar(value=self.speed)
        speed_scale = tk.Scale(speed_frame, from_=100, to=2000, orient=tk.HORIZONTAL,
                              variable=self.speed_var, command=self.update_speed,
                              length=200, bg='#eff6ff', font=('Arial', 9))
        speed_scale.pack(side=tk.LEFT, padx=5)
        self.speed_label = tk.Label(speed_frame, text=f"{self.speed}ms", 
                                    font=('Arial', 10), bg='#eff6ff')
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Step slider
        slider_frame = tk.Frame(main_frame, bg='#eff6ff')
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.step_label = tk.Label(slider_frame, text="Step 0 / 0", 
                                   font=('Arial', 10), bg='#eff6ff')
        self.step_label.pack(side=tk.LEFT, padx=10)
        
        self.step_var = tk.IntVar(value=0)
        self.step_slider = tk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                    variable=self.step_var, command=self.on_slider_change,
                                    length=900, bg='#eff6ff', showvalue=False)
        self.step_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=1200, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 15))
        
        # Metrics
        metrics_frame = tk.Frame(main_frame, bg='#eff6ff')
        metrics_frame.pack(pady=(0, 15))
        
        zeros_frame = tk.Frame(metrics_frame, bg='#dbeafe', relief=tk.SOLID, borderwidth=2)
        zeros_frame.pack(side=tk.LEFT, padx=10)
        self.zeros_label = tk.Label(zeros_frame, text="0", font=('Arial', 32, 'bold'),
                                    bg='#dbeafe', fg='#1e40af', width=8)
        self.zeros_label.pack(padx=20, pady=(10, 5))
        tk.Label(zeros_frame, text="Number of 0s", font=('Arial', 10),
                bg='#dbeafe', fg='#6b7280').pack(padx=20, pady=(0, 10))
        
        ones_frame = tk.Frame(metrics_frame, bg='#e9d5ff', relief=tk.SOLID, borderwidth=2)
        ones_frame.pack(side=tk.LEFT, padx=10)
        self.ones_label = tk.Label(ones_frame, text="0", font=('Arial', 32, 'bold'),
                                   bg='#e9d5ff', fg='#6b21a8', width=8)
        self.ones_label.pack(padx=20, pady=(10, 5))
        tk.Label(ones_frame, text="Number of 1s", font=('Arial', 10),
                bg='#e9d5ff', fg='#6b7280').pack(padx=20, pady=(0, 10))
        
        # Current step display
        step_display_frame = tk.Frame(main_frame, bg='white', relief=tk.SOLID, borderwidth=2)
        step_display_frame.pack(fill=tk.BOTH, pady=(0, 15))
        
        self.step_title = tk.Label(step_display_frame, text="Current Step: -", 
                                   font=('Arial', 11, 'bold'), bg='white', fg='#374151')
        self.step_title.pack(anchor=tk.W, padx=10, pady=10)
        
        # Canvas for token display
        canvas_container = tk.Frame(step_display_frame, bg='white')
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.token_canvas = tk.Canvas(canvas_container, bg='white', height=100, highlightthickness=0)
        self.token_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        h_scrollbar = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.token_canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.token_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Legend
        legend_frame = tk.Frame(main_frame, bg='#f9fafb', relief=tk.SOLID, borderwidth=1)
        legend_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(legend_frame, text="Token Legend", font=('Arial', 11, 'bold'),
                bg='#f9fafb', fg='#374151').pack(anchor=tk.W, padx=10, pady=10)
        
        legend_items = tk.Frame(legend_frame, bg='#f9fafb')
        legend_items.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        for value, info in TOKEN_MAP.items():
            item = tk.Frame(legend_items, bg='#f9fafb')
            item.pack(side=tk.LEFT, padx=10)
            
            token_box = tk.Label(item, text=info['name'], font=('Courier', 10, 'bold'),
                               bg=info['color'], fg=info['text'], relief=tk.SOLID,
                               borderwidth=2, padx=8, pady=4)
            token_box.pack(side=tk.LEFT, padx=(0, 5))
            
            tk.Label(item, text=info['desc'], font=('Arial', 9), 
                    bg='#f9fafb', fg='#6b7280').pack(side=tk.LEFT)
        
        # Data token legend
        data_item = tk.Frame(legend_items, bg='#f9fafb')
        data_item.pack(side=tk.LEFT, padx=10)
        data_box = tk.Label(data_item, text="0-1", font=('Courier', 10, 'bold'),
                           bg='#dbeafe', fg='#1e40af', relief=tk.SOLID,
                           borderwidth=2, padx=8, pady=4)
        data_box.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(data_item, text="Data", font=('Arial', 9),
                bg='#f9fafb', fg='#6b7280').pack(side=tk.LEFT)
        
        # Static diagram frame
        static_frame = tk.Frame(main_frame, bg='white', relief=tk.SOLID, borderwidth=2)
        static_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(static_frame, text="Complete Denoising Process", 
                font=('Arial', 14, 'bold'), bg='white', fg='#374151').pack(anchor=tk.W, padx=10, pady=10)
        
        # Create scrollable canvas for static diagram
        static_container = tk.Frame(static_frame, bg='white')
        static_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.static_canvas = tk.Canvas(static_container, bg='white', highlightthickness=0)
        v_scrollbar = tk.Scrollbar(static_container, orient=tk.VERTICAL, command=self.static_canvas.yview)
        h_scrollbar_static = tk.Scrollbar(static_container, orient=tk.HORIZONTAL, command=self.static_canvas.xview)
        
        self.static_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar_static.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar_static.pack(side=tk.BOTTOM, fill=tk.X)
        self.static_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def update_speed(self, value):
        self.speed = int(value)
        self.speed_label.config(text=f"{self.speed}ms")
        
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select denoising file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                self.parse_file(content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def parse_file(self, content):
        lines = content.strip().split('\n')
        self.steps = []
        for line in lines:
            tokens = [int(x) for x in line.strip().split()]
            self.steps.append(tokens)
        
        if self.steps:
            self.current_step = 0
            self.step_slider.configure(to=len(self.steps) - 1)
            self.step_var.set(0)
            self.play_btn.config(state=tk.NORMAL)
            self.update_display()
            self.draw_static_diagram()
            
    def get_token_display(self, value):
        if value in TOKEN_MAP:
            return TOKEN_MAP[value]
        return {
            'name': str(value),
            'color': '#dbeafe',
            'border': '#93c5fd',
            'text': '#1e40af',
            'desc': 'Data Token'
        }
        
    def count_zeros_ones(self, step):
        data_tokens = [t for t in step if t in [0, 1]]
        zeros = sum(1 for t in data_tokens if t == 0)
        ones = sum(1 for t in data_tokens if t == 1)
        return zeros, ones
        
    def check_goal_achieved(self, step):
        zeros, ones = self.count_zeros_ones(step)
        if zeros != ones:
            return False
        
        data_tokens = [t for t in step if t in [0, 1]]
        seen_one = False
        for token in data_tokens:
            if token == 1:
                seen_one = True
            if token == 0 and seen_one:
                return False
        return True
        
    def update_display(self):
        if not self.steps:
            return
            
        step = self.steps[self.current_step]
        
        # Update step label
        self.step_label.config(text=f"Step {self.current_step + 1} / {len(self.steps)}")
        
        # Update progress
        progress_value = ((self.current_step + 1) / len(self.steps)) * 100
        self.progress['value'] = progress_value
        
        # Update metrics
        zeros, ones = self.count_zeros_ones(step)
        self.zeros_label.config(text=str(zeros))
        self.ones_label.config(text=str(ones))
        
        # Update step title
        if self.current_step == 0:
            step_text = "Maximum Noise"
        elif self.current_step == len(self.steps) - 1:
            step_text = "Clean Signal"
        else:
            step_text = f"Step {self.current_step}"
        self.step_title.config(text=f"Current Step: {step_text}")
        
        # Draw tokens
        self.token_canvas.delete("all")
        x_offset = 10
        y_offset = 10
        token_width = 60
        token_height = 60
        gap = 5
        
        for i, token in enumerate(step):
            display = self.get_token_display(token)
            
            # Draw rectangle
            x1 = x_offset + i * (token_width + gap)
            y1 = y_offset
            x2 = x1 + token_width
            y2 = y1 + token_height
            
            self.token_canvas.create_rectangle(x1, y1, x2, y2, 
                                              fill=display['color'],
                                              outline=display['border'],
                                              width=2)
            
            # Draw text
            self.token_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                         text=display['name'],
                                         font=('Courier', 11, 'bold'),
                                         fill=display['text'])
        
        # Update scroll region
        self.token_canvas.configure(scrollregion=self.token_canvas.bbox("all"))
        
        # Check if goal achieved
        if self.current_step == len(self.steps) - 1 and self.check_goal_achieved(step):
            self.show_success()
            
    def draw_static_diagram(self):
        if not self.steps:
            return
            
        self.static_canvas.delete("all")
        
        token_width = 60
        token_height = 40
        gap = 5
        row_gap = 5
        label_width = 50
        
        for step_idx, step in enumerate(self.steps):
            y_offset = 10 + step_idx * (token_height + row_gap)
            
            # Draw step label
            self.static_canvas.create_text(label_width / 2, y_offset + token_height / 2,
                                          text=str(step_idx),
                                          font=('Courier', 9),
                                          fill='#6b7280')
            
            # Draw tokens
            for token_idx, token in enumerate(step):
                display = self.get_token_display(token)
                
                x1 = label_width + token_idx * (token_width + gap)
                y1 = y_offset
                x2 = x1 + token_width
                y2 = y1 + token_height
                
                # Highlight current step
                outline_width = 4 if step_idx == self.current_step else 2
                outline_color = '#4f46e5' if step_idx == self.current_step else display['border']
                
                self.static_canvas.create_rectangle(x1, y1, x2, y2,
                                                   fill=display['color'],
                                                   outline=outline_color,
                                                   width=outline_width)
                
                self.static_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                              text=display['name'],
                                              font=('Courier', 9, 'bold'),
                                              fill=display['text'])
        
        self.static_canvas.configure(scrollregion=self.static_canvas.bbox("all"))
        
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self.play_animation()
            
    def play_animation(self):
        if not self.is_playing:
            return
            
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.step_var.set(self.current_step)
            self.update_display()
            self.draw_static_diagram()
            self.play_job = self.root.after(self.speed, self.play_animation)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            
    def reset(self):
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.current_step = 0
        self.step_var.set(0)
        if self.steps:
            self.update_display()
            self.draw_static_diagram()
            
    def on_slider_change(self, value):
        if self.is_playing:
            self.toggle_play()
        self.current_step = int(value)
        self.update_display()
        self.draw_static_diagram()
        
    def show_success(self):
        success_window = tk.Toplevel(self.root)
        success_window.title("Goal Achieved!")
        success_window.geometry("600x400")
        success_window.configure(bg='#10b981')
        
        tk.Label(success_window, text="🎉", font=('Arial', 72), 
                bg='#10b981').pack(pady=30)
        
        tk.Label(success_window, text="Goal Achieved!", 
                font=('Arial', 36, 'bold'), bg='#10b981', fg='white').pack(pady=10)
        
        tk.Label(success_window, 
                text="The model successfully unmasked the string with\nequal 0s and 1s, all 0s before 1s!",
                font=('Arial', 14), bg='#10b981', fg='white', justify=tk.CENTER).pack(pady=20)
        
        tk.Button(success_window, text="Close", command=success_window.destroy,
                 font=('Arial', 14), bg='#4f46e5', fg='white',
                 padx=30, pady=10, relief=tk.FLAT, cursor='hand2').pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingVisualizer(root)
    root.mainloop()