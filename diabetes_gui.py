import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict


class FuzzyDiabetesSystem:
    def __init__(self):
        self.ranges = {
            'blood_sugar': (0, 301, 1),
            'bmi': (0, 51, 1),
            'age': (0, 101, 1),
            'risk': (0, 101, 1)
        }

        self.blood_sugar = ctrl.Antecedent(np.arange(*self.ranges['blood_sugar']), 'blood_sugar')
        self.bmi = ctrl.Antecedent(np.arange(*self.ranges['bmi']), 'bmi')
        self.age = ctrl.Antecedent(np.arange(*self.ranges['age']), 'age')
        self.risk = ctrl.Consequent(np.arange(*self.ranges['risk']), 'risk')

        self._setup_membership_functions()
        self._setup_rules()

        self.control_system = ctrl.ControlSystem(self.rules)
        self.diagnosis_sim = ctrl.ControlSystemSimulation(self.control_system)

    def _setup_membership_functions(self):
        self.blood_sugar['low'] = fuzz.trimf(self.blood_sugar.universe, [0, 0, 80])
        self.blood_sugar['normal'] = fuzz.trimf(self.blood_sugar.universe, [70, 90, 110])
        self.blood_sugar['high'] = fuzz.trimf(self.blood_sugar.universe, [100, 125, 150])
        self.blood_sugar['very_high'] = fuzz.trapmf(self.blood_sugar.universe, [140, 160, 300, 300])

        self.bmi['underweight'] = fuzz.trimf(self.bmi.universe, [0, 0, 18.5])
        self.bmi['normal'] = fuzz.trimf(self.bmi.universe, [18, 22, 25])
        self.bmi['overweight'] = fuzz.trimf(self.bmi.universe, [24, 27, 30])
        self.bmi['obese'] = fuzz.trapmf(self.bmi.universe, [29, 32, 50, 50])

        self.age['young'] = fuzz.trimf(self.age.universe, [0, 0, 30])
        self.age['middle_aged'] = fuzz.trimf(self.age.universe, [25, 45, 60])
        self.age['elderly'] = fuzz.trapmf(self.age.universe, [50, 65, 100, 100])

        self.risk['low'] = fuzz.trimf(self.risk.universe, [0, 0, 40])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [30, 50, 70])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [60, 100, 100])

    def _setup_rules(self):
        self.rules = [
            ctrl.Rule(self.blood_sugar['normal'] & self.bmi['normal'] & self.age['young'], self.risk['low']),
            ctrl.Rule(self.blood_sugar['normal'] & self.bmi['normal'] & self.age['middle_aged'], self.risk['low']),
            ctrl.Rule(self.blood_sugar['high'] & self.bmi['overweight'], self.risk['medium']),
            ctrl.Rule(self.blood_sugar['normal'] & self.bmi['obese'], self.risk['medium']),
            ctrl.Rule(self.blood_sugar['high'] & self.age['elderly'], self.risk['medium']),
            ctrl.Rule(self.blood_sugar['very_high'] & self.bmi['obese'], self.risk['high']),
            ctrl.Rule(self.blood_sugar['very_high'] & self.age['elderly'], self.risk['high']),
            ctrl.Rule(self.blood_sugar['high'] & self.bmi['obese'] & self.age['elderly'], self.risk['high'])
        ]

    def assess_risk(self, blood_sugar: float, bmi: float, age: float) -> float:
        self.diagnosis_sim.input['blood_sugar'] = blood_sugar
        self.diagnosis_sim.input['bmi'] = bmi
        self.diagnosis_sim.input['age'] = age

        try:
            self.diagnosis_sim.compute()
            return self.diagnosis_sim.output['risk']
        except Exception as e:
            print(f"Error: {e}")
            return 0


class DiabetesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Diabetes Risk Assessment System")
        self.root.geometry("1200x800")

        self.fuzzy_system = FuzzyDiabetesSystem()

        self._setup_styles()
        self._create_widgets()
        self._setup_layout()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 14))
        style.configure('Input.TLabel', font=('Helvetica', 10))
        style.configure('Action.TButton', font=('Helvetica', 10, 'bold'))

    def _create_widgets(self):
        self.input_frame = ttk.LabelFrame(self.root, text="Patient Data", padding=15)

        self.inputs = {}
        for label, unit in [("Blood Sugar", "mg/dL"), ("BMI", "kg/m²"), ("Age", "years")]:
            container = ttk.Frame(self.input_frame)
            ttk.Label(container, text=f"{label} ({unit}):", style='Input.TLabel').pack(side=tk.LEFT)
            entry = ttk.Entry(container, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            self.inputs[label.lower().replace(" ", "_")] = entry
            container.pack(pady=5)

        self.button_frame = ttk.Frame(self.input_frame)
        ttk.Button(self.button_frame, text="Assess Risk", command=self.assess_risk, style='Action.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Update Visualization", command=self.update_visualization, style='Action.TButton').pack(side=tk.LEFT, padx=5)
        self.button_frame.pack(pady=10)

        self.result_frame = ttk.LabelFrame(self.root, text="Assessment Results", padding=15)
        self.result_label = ttk.Label(self.result_frame, text="Awaiting assessment...", style='Result.TLabel')
        self.result_label.pack(pady=10)

        self.viz_frame = ttk.LabelFrame(self.root, text="Fuzzy Logic Visualization", padding=15)
        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.result_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.viz_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")

    def assess_risk(self):
        try:
            values = {key: float(entry.get()) for key, entry in self.inputs.items()}
            if not self._validate_inputs(values):
                return

            risk_level = self.fuzzy_system.assess_risk(**values)
            self._update_result_display(risk_level)
            self.update_visualization()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields!")

    def _validate_inputs(self, values: Dict[str, float]) -> bool:
        ranges = {
            'blood_sugar': (0, 300),
            'bmi': (0, 50),
            'age': (0, 100)
        }

        for key, (min_val, max_val) in ranges.items():
            if not min_val <= values[key] <= max_val:
                messagebox.showerror("Invalid Input", f"{key.replace('_', ' ').title()} must be between {min_val} and {max_val}")
                return False
        return True

    def _update_result_display(self, risk_level: float):
        category = "Low" if risk_level < 40 else "Medium" if risk_level < 70 else "High"
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        self.result_label.configure(text=f"Risk Level: {risk_level:.1f}%\nCategory: {category} Risk", foreground=colors[category])

    def update_visualization(self):
        try:
            values = {key: float(entry.get()) for key, entry in self.inputs.items()}
            self._plot_membership_functions(values)
        except ValueError:
            self._plot_membership_functions()

    def _plot_membership_functions(self, current_values: Dict[str, float] = None):
        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        variables = [('blood_sugar', self.fuzzy_system.blood_sugar), ('bmi', self.fuzzy_system.bmi),
                     ('age', self.fuzzy_system.age), ('risk', self.fuzzy_system.risk)]

        for ax, (var_name, var) in zip(axes.flat, variables):
            for term in var.terms:
                ax.plot(var.universe, var[term].mf, label=term)
            if current_values and var_name in current_values:
                ax.axvline(current_values[var_name], color='black', linestyle='--')
            ax.set_title(var_name.replace('_', ' ').title())
            ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesApp(root)
    root.mainloop()
