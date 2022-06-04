"""User interface for running video detection and tracking inference."""

# pylint: disable=line-too-long
# pylint: disable=wildcard-import
from functools import partial
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from typing import List, Tuple


class InferenceUI:
    """The user interface for running tracking and inference."""
    file_entry_width = 24

    def __init__(self, root):
        root.title("Deepsea-Detector UI")
        
        label_padding = ("3 3 3 3")

        # Create frame widget
        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)

        ### ===================
        ### INPUT
        ### ===================
        # Create a labelframe for the input
        input_frame = ttk.Labelframe(mainframe, text="Input", padding="3 3 12 12")
        input_frame.grid(column=1, row=1, sticky=(N, E, W, S))  # Defines location on the main UI grid
        input_frame.columnconfigure(2, weight=1)  # Makes the entry fill up any available space

        # Video input label, entry, and button
        ttk.Label(input_frame, text="Video Input:", padding=label_padding).grid(column=1, row=1, sticky=NW)
        self.video_in = StringVar()  # String variable that can be read/accessed by Tk
        video_in_entry = ttk.Entry(input_frame, width=InferenceUI.file_entry_width, textvariable=self.video_in)
        video_in_entry.grid(column=2, row=1, sticky=(W, E))
        # Defines button that opens a file browser when pressed
        video_in_button = ttk.Button(input_frame, text="Browse", command=partial(self.browse, self.video_in))
        video_in_button.grid(column=3, row=1, sticky=W)

        ### ===================
        ### OUTPUT
        ### ===================
        out_frame = ttk.LabelFrame(mainframe, text="Output", padding="3 3 12 12")
        out_frame.grid(column=1, row=2, sticky=(N, E, W, S))
        out_frame.columnconfigure(2, weight=1)

        # Video output label, entry, and button
        ttk.Label(out_frame, text="Video Output:", padding=label_padding)\
                 .grid(column=1, row=1, sticky=NW)
        self.video_out = StringVar(value="out.mp4")  # String variable that can be read/accessed by Tk
        video_out_entry = ttk.Entry(out_frame,
                                    width=InferenceUI.file_entry_width,
                                    textvariable=self.video_out)
        video_out_entry.grid(column=2, row=1, sticky=(W, E))
        # Button command, when pressed asks for save location of MP4 file
        video_out_button_cmd = partial(self.save_as, self.video_out, self.video_out.get(), [("MP4", ".mp4")])
        video_out_button = ttk.Button(out_frame, text="Browse", command=video_out_button_cmd)
        video_out_button.grid(column=3, row=1, sticky=W)
        ttk.Label(out_frame, text="Saves a video showing identifications and inferences made.")\
                .grid(column=2, row=2, sticky=W)

        # CSV output label, entry, and button
        ttk.Label(out_frame, text="CSV Output:", padding=label_padding)\
                 .grid(column=1, row=3, sticky=NW)
        self.csv_out = StringVar(value="out.csv")  # String variable that can be read/accessed by Tk
        csv_out_entry = ttk.Entry(out_frame,
                                  width=InferenceUI.file_entry_width,
                                  textvariable=self.csv_out)
        csv_out_entry.grid(column=2, row=3, sticky=(W, E))
        # Gets save location for CSV file
        csv_out_button_cmd = partial(self.save_as, self.csv_out, self.csv_out.get(), [("Comma Separated Value File", ".csv")])
        csv_out_button = ttk.Button(out_frame, text="Browse", command=csv_out_button_cmd)
        csv_out_button.grid(column=3, row=3, sticky=W)
        ttk.Label(out_frame, text="Comma separated value file of all detected organisms.").grid(column=2, row=4, sticky=W)

        ### ===================
        ### MODEL CONFIGURATION
        ### ===================
        ml_frame = ttk.Labelframe(mainframe, text="ML Model Configuration", padding="3 3 12 12")
        ml_frame.grid(column=1, row=3, sticky=(N, E, W, S))
        ml_frame.columnconfigure(2, weight=1)

        # Model Weights (.pt)
        ttk.Label(ml_frame, text="YOLO Model:", padding=label_padding).grid(column=1, row=1, sticky=NW)
        self.ml_model_weights = StringVar()
        ttk.Entry(ml_frame, width=InferenceUI.file_entry_width, textvariable=self.ml_model_weights)\
                 .grid(column=2, row=1, sticky=(W, E))
        ml_browse_command = partial(self.browse, self.ml_model_weights, ".pt", [("PyTorch Model", ".pt")])
        ml_model_button = ttk.Button(ml_frame, text="Browse", command=ml_browse_command)
        ml_model_button.grid(column=3, row=1, sticky=W)
        ttk.Label(ml_frame, text="The YOLOv5 detection model to use.").grid(column=2, row=2, sticky=W)

        # Period/Stride Entry
        ttk.Label(ml_frame, text="Period:", padding=label_padding).grid(column=1, row=3, sticky=NW)
        self.period = IntVar(value=3)  # String variable that can be read/accessed by Tk
        ttk.Entry(ml_frame, width=3, textvariable=self.period)\
                 .grid(column=2, row=3, sticky=(W))
        ttk.Label(ml_frame, text="How often, in frames, to run the detection algorithm. \nHigher values can speed up processing but may \ndecrease accuracy. (default is 3)")\
                  .grid(column=2, row=4, sticky=W)

        ### RUN INFERENCE
        ttk.Button(mainframe, text="Run Inference", width=24).grid(column="1", row="4", sticky=(E))

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        root.bind("<Return>", self.run_inference)

    def browse(self, target: StringVar, default_extension:str="", filetypes: List[str]=list()):
        "Opens a file dialog and sets the given target StringVar to the selected file path."
        file_path = filedialog.askopenfilename(filetypes=filetypes,
                                               defaultextension=default_extension)
        target.set(file_path)

    def save_as(self, target: StringVar, default_path: str, extension:List[Tuple[str, str]]=list()):
        "Opens a file save dialog and sets the given StringVar to that file path."
        file_path = filedialog.asksaveasfilename(initialfile=default_path, filetypes=extension)
        target.set(file_path)

    def run_inference(self):
        "Starts inference calculations, using the input settings."
        pass


root = Tk()
InferenceUI(root)
root.mainloop()
