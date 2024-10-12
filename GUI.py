import customtkinter as ctk
from tkinter import Canvas, Toplevel
from PIL import Image, ImageGrab
import time
from add_to_csv import process_images
from predict import predict_using_images
import model
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import sys

sys.dont_write_bytecode = True


class DrawingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Drawing Application")
        self.geometry("240x380")
        self.resizable(False, False)

        # Title
        self.title_label = ctk.CTkLabel(self, text="MNIST + KNN", font=("system", 30))
        self.title_label.pack(pady=20)

        # Canvas
        self.canvas = Canvas(
            self,
            bg="black",
            width=170,
            height=170,
            highlightcolor="black",
            highlightbackground="black",
            highlightthickness=10,
        )
        self.canvas.pack(padx=20)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.button_frame = ctk.CTkFrame(self, corner_radius=0)
        self.button_frame.pack(pady=20)

        self.buttons = []
        button_commands = {
            "Clear": self.clear_canvas,
            "Test": self.save_canvas_as_image,
            "Retrain": self.train,
        }
        button_frame_top = ctk.CTkFrame(self.button_frame, fg_color="#2b2b2b")
        button_frame_top.pack(pady=5)
        for text, command in list(button_commands.items())[:2]:
            button = ctk.CTkButton(
                button_frame_top,
                text=text,
                command=command,
                width=85,
                fg_color="#d0d0d0",
                text_color="#000000",
                hover_color="#a0c0ff",
                font=("system", 18),
                corner_radius=0,
            )
            button.pack(side="left", padx=5)
            self.buttons.append(button)

        for text, command in list(button_commands.items())[2:]:
            button = ctk.CTkButton(
                self.button_frame,
                text=text,
                command=command,
                width=180,
                fg_color="#d0d0d0",
                text_color="#000000",
                hover_color="#a0c0ff",
                font=("system", 18),
                corner_radius=0,
            )
            button.pack(pady=5, padx=5)
            self.buttons.append(button)

        self.drawing_color = "white"

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(
            x1, y1, x2, y2, fill=self.drawing_color, outline=self.drawing_color
        )

    def clear_canvas(self):
        self.canvas.delete("all")

    def train(self):
        self.config(cursor="wait")
        result = model.train()
        self.config(cursor="arrow")
        print(result)

    def save_canvas_as_image(self):
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + 190
        y1 = y0 + 190
        img = ImageGrab.grab().crop((x0, y0, x1, y1))
        img = img.resize((28, 28))
        now = time.time()
        img.save(f"images\\{now}.png")
        result = predict_using_images()
        self.display_results(result)

    def display_results(self, result):
        predictions = list(result.values())[0]
        maxProb = max(predictions)
        top = ctk.CTkToplevel(self)
        filename = list(result.keys())[0]

        top.title(filename)
        top.grab_set()
        for key, values in result.items():
            fig = Figure(figsize=(5, 4), dpi=100)

            fig.patch.set_facecolor("#242424")

            ax = fig.add_subplot(111)
            ax.patch.set_facecolor("black")
            ax.spines["bottom"].set_color("#d0d0d0")
            ax.spines["left"].set_color("#d0d0d0")

            ax.bar(range(len(values)), values, width=0.5, color="#d0d0d0")

            for i, value in enumerate(values):
                ax.text(
                    i,
                    value + 2,
                    f"{value:.0f}%",
                    ha="center",
                    color="#d0d0d0",
                    fontsize=8,
                )
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(range(len(values)))
            ax.tick_params(axis="x", colors="#d0d0d0", labelsize=10)
            ax.tick_params(axis="y", colors="#d0d0d0", labelsize=10)
            ax.set_ylim(0, 110)
            ax.set_title(
                f"Prediction: {predictions.index(maxProb)}",
                color="white",
                fontsize=12,
            )

            canvas = FigureCanvasTkAgg(fig, master=top)

            canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

            def get_actual_answer():
                actual_answer_window = ctk.CTkToplevel(top)

                actual_answer_window.title("Enter Actual Answer")

                actual_answer_window.grab_set()  # Make the window modal

                buttons_frame = ctk.CTkFrame(actual_answer_window)

                buttons_frame.pack(pady=10)

                buttons = []
                for i in range(10):
                    button = ctk.CTkButton(
                        buttons_frame,
                        text=str(i),
                        width=40,
                        command=lambda i=i: submit_answer(i),
                        font=("System", 10),
                        corner_radius=0,
                        fg_color="#d0d0d0",
                        text_color="black",
                        hover_color="#a0a0a0",
                    )

                    button.grid(row=i // 5, column=i % 5, padx=5, pady=5)

                    buttons.append(button)

                def submit_answer(actual_answer):
                    with open("new_data.json") as f:
                        data = json.load(f)

                    data[list(result.keys())[0]] = actual_answer

                    with open("new_data.json", "w") as f:
                        json.dump(data, f, indent=4)
                    image_directory = "predictions/"
                    json_file_path = "new_data.json"
                    output_csv_path = "externals.csv"
                    process_images(image_directory, json_file_path, output_csv_path)

                    actual_answer_window.destroy()
                    actual_answer_window.grab_release()
                    top.destroy()

            button = ctk.CTkButton(
                top,
                text="Wrong Prediction",
                command=get_actual_answer,
                corner_radius=0,
                font=("System", 10),
                fg_color="#d0d0d0",
                text_color="black",
                hover_color="#f0b0b0",
            )

            button.pack(pady=10)


if __name__ == "__main__":
    app = DrawingApp()
    app.mainloop()
