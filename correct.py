import argparse
import json
import tkinter as tk
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageTk
import skimage


class App(tk.Tk):
    POINT_RADIUS = 5

    def __init__(
        self,
        gaze,
        image,
        title,
        src_points=None,
        dst_points=None,
        margin=100,
        scale=1.0,
        fixation_cross=None,
        vertical=False,
    ):
        super().__init__()
        self.gaze = gaze
        self.image = PIL.ImageTk.PhotoImage(
            image.resize((round(image.width * scale), round(image.height * scale)))
        )

        self.margin = margin
        self.scale = scale
        self.fixation_cross = fixation_cross
        self.vertical = vertical

        self.simplified_gaze = self._simplify_gaze(10)

        # Set default source and destination points if not provided
        if src_points is None:
            src_points = [
                (0, 0),
                (image.width / 2, 0),
                (image.width, 0),
                (0, image.height),
                (image.width / 2, image.height),
                (image.width, image.height),
            ]
        if dst_points is None:
            dst_points = src_points.copy()
        self.src_points = src_points
        self.dst_points = dst_points

        # State variables related to user input
        self.hover_point_index = None
        self.dragging_point_index = None
        self.history = []
        self.action = None

        self.title(title)
        self.resizable(False, False)
        width = self.image.width() + 2 * self.margin
        height = self.image.height() + 2 * self.margin
        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            background="white",
        )
        self.canvas.pack()

        # Center the window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Mouse and keyboard bindings
        self.canvas.bind("<Button-1>", self._on_left_mouse_down)
        self.canvas.bind("<Button-3>", self._on_right_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_mouse_up)
        self.canvas.bind("<Motion>", self._on_left_mouse_move)
        self.canvas.bind("<B1-Motion>", self._on_left_mouse_drag)
        self.bind("<Control-z>", lambda _: self._undo())
        self.bind("<Right>", lambda _: self._exit("next"))
        self.bind("<Left>", lambda _: self._exit("previous"))
        self.bind("<Escape>", lambda _: self._exit("exit"))

        self._draw()
        self.focus_force()
        self.mainloop()

    def get_transform(self):
        transform = skimage.transform.ThinPlateSplineTransform()
        src = np.array(self.src_points)
        dst = np.array(self.dst_points)
        transform.estimate(src, dst)
        return transform

    def _gaze_to_window_coords(self, x, y):
        return x * self.scale + self.margin, y * self.scale + self.margin

    def _window_to_gaze_coords(self, x, y):
        return (x - self.margin) / self.scale, (y - self.margin) / self.scale

    def _simplify_gaze(self, n=10):
        gaze = self.gaze[["pixel_x", "pixel_y"]]
        gaze = gaze.groupby(gaze.index // n).mean().reset_index(drop=True)
        return gaze

    def _on_left_mouse_down(self, event):
        if self.hover_point_index is not None:
            self._store_history()
            self.dragging_point_index = self.hover_point_index
        else:
            self._store_history()
            self.src_points.append(self._window_to_gaze_coords(event.x, event.y))
            self.dst_points.append(self._window_to_gaze_coords(event.x, event.y))
            self.dragging_point_index = len(self.dst_points) - 1
        self._draw()

    def _on_right_mouse_down(self, event):
        if self.hover_point_index is not None:
            self._store_history()
            del self.src_points[self.hover_point_index]
            del self.dst_points[self.hover_point_index]
            self.dragging_point_index = None
        self._draw()

    def _on_left_mouse_up(self, event):
        self.dragging_point_index = None
        self._draw()

    def _on_left_mouse_move(self, event):
        hover_point_index_before = self.hover_point_index
        for i, (x, y) in enumerate(self.dst_points):
            x, y = self._gaze_to_window_coords(x, y)
            dx = event.x - x
            dy = event.y - y
            if dx * dx + dy * dy < self.POINT_RADIUS**2:
                self.hover_point_index = i
                break
        else:
            self.hover_point_index = None

        if hover_point_index_before != self.hover_point_index:
            self._draw()

    def _on_left_mouse_drag(self, event):
        if self.dragging_point_index is not None:
            self.hover_point_index = None
            x, y = self._window_to_gaze_coords(event.x, event.y)
            if self.vertical:
                x = self.dst_points[self.dragging_point_index][0]
            self.dst_points[self.dragging_point_index] = (x, y)
            self._draw()

    def _undo(self):
        if self.history:
            self.src_points, self.dst_points = self.history.pop()
            self._draw()

    def _store_history(self):
        self.history.append((self.src_points.copy(), self.dst_points.copy()))

    def _draw(self):
        self.canvas.delete("all")

        self.canvas.create_image(
            *self._gaze_to_window_coords(0, 0), anchor=tk.NW, image=self.image
        )

        if self.fixation_cross:
            x, y = self._gaze_to_window_coords(*self.fixation_cross)
            x1 = x - 10
            y1 = y - 10
            x2 = x + 10
            y2 = y + 10
            self.canvas.create_line(x1, y, x2, y, fill="red", width=2)
            self.canvas.create_line(x, y1, x, y2, fill="red", width=2)

        gaze = self.simplified_gaze
        gaze = pd.concat([gaze, gaze.shift(-1).add_prefix("next_")], axis=1)

        transform = self.get_transform()
        transformed_gaze = gaze.copy()
        transformed_gaze[["pixel_x", "pixel_y"]] = transform(
            gaze[["pixel_x", "pixel_y"]]
        )
        transformed_gaze[["next_pixel_x", "next_pixel_y"]] = transform(
            gaze[["next_pixel_x", "next_pixel_y"]]
        )

        # Transform replaces NA with -1, so we need to set them back to NA
        transformed_gaze.loc[gaze["pixel_x"].isna(), "pixel_x"] = None
        transformed_gaze.loc[gaze["pixel_y"].isna(), "pixel_y"] = None
        transformed_gaze.loc[gaze["next_pixel_x"].isna(), "next_pixel_x"] = None
        transformed_gaze.loc[gaze["next_pixel_y"].isna(), "next_pixel_y"] = None

        for _, row in transformed_gaze.iterrows():
            x = row["pixel_x"]
            y = row["pixel_y"]
            next_x = row["next_pixel_x"]
            next_y = row["next_pixel_y"]
            self.canvas.create_line(
                *self._gaze_to_window_coords(x, y),
                *self._gaze_to_window_coords(next_x, next_y),
                fill="black",
                width=2,
            )

        for i, ((src_x, src_y), (dst_x, dst_y)) in enumerate(
            zip(self.src_points, self.dst_points)
        ):
            src_x, src_y = self._gaze_to_window_coords(src_x, src_y)
            dst_x, dst_y = self._gaze_to_window_coords(dst_x, dst_y)
            self.canvas.create_oval(
                src_x - self.POINT_RADIUS,
                src_y - self.POINT_RADIUS,
                src_x + self.POINT_RADIUS,
                src_y + self.POINT_RADIUS,
                fill="lightblue",
                outline="lightblue",
            )
            self.canvas.create_line(
                src_x,
                src_y,
                dst_x,
                dst_y,
                fill="lightblue",
            )
            self.canvas.create_oval(
                dst_x - self.POINT_RADIUS,
                dst_y - self.POINT_RADIUS,
                dst_x + self.POINT_RADIUS,
                dst_y + self.POINT_RADIUS,
                fill=(
                    "red"
                    if i == self.hover_point_index or i == self.dragging_point_index
                    else "blue"
                ),
            )

    def _exit(self, action):
        self.action = action
        self.destroy()


def main(gaze_path, vertical):
    gaze_path = Path(gaze_path)
    gaze = pd.read_csv(gaze_path)
    stimuli = list(gaze["stimulus"].unique())

    transforms_path = gaze_path.with_suffix(".transforms.json")
    if transforms_path.exists():
        with open(transforms_path) as f:
            transforms = json.load(f)
    else:
        transforms = {stimulus: (None, None) for stimulus in stimuli}

    stimulus_index = 0
    while True:
        stimulus = stimuli[stimulus_index]
        stimulus_gaze = gaze[gaze["stimulus"] == stimulus]
        stimulus_filename = f"{stimulus}.word.png"
        image = PIL.Image.open(gaze_path.parent.parent / "stimuli" / stimulus_filename)
        src_points, dst_points = transforms[stimulus]
        app = App(
            stimulus_gaze,
            image,
            stimulus,
            src_points=src_points,
            dst_points=dst_points,
            scale=0.5,
            fixation_cross=(15, 80),
            vertical=vertical,
        )
        if app.action == "next":
            transforms[stimulus] = (app.src_points, app.dst_points)
            stimulus_index += 1
            if stimulus_index >= len(stimuli):
                break
        elif app.action == "previous":
            transforms[stimulus] = (app.src_points, app.dst_points)
            stimulus_index -= 1
            if stimulus_index < 0:
                break
        else:
            transforms[stimulus] = (app.src_points, app.dst_points)
            break

    with open(transforms_path, "w") as f:
        json.dump(transforms, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gaze_path",
        type=Path,
        help="Path to the gaze data CSV file.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Only allow vertical transformations.",
    )
    args = parser.parse_args()
    main(args.gaze_path, args.vertical)
