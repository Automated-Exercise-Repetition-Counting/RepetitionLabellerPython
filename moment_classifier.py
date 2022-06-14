import cv2
import os
import numpy as np
import pandas as pd
import moment_map


# User Input
ABSOLUTE_VIDEO_PATH = "G:\Shared drives\P4P\Data Collection\Squats\IMG_2167.MOV"
OUTPUT_DIR = "output"
DEFAULT_PLAYBACK_FPS = 30
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT

# Constants
INITIAL_MOMENT = 0
OPEN_CV_COLOUR_MAP = [0, 1, 3, 5, 6, 7, 8, 10, 11]

video_name = os.path.basename(ABSOLUTE_VIDEO_PATH).split(".")[0]
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"{video_name}_labels.csv")

output_images_name = video_name + "_images_np"
absolute_images_np_path = os.path.join(OUTPUT_DIR, output_images_name) + ".npy"


def np_array_from_images():
    if not os.path.exists(absolute_images_np_path):
        print("NP Array does not exist, creating...")

        cap = cv2.VideoCapture(ABSOLUTE_VIDEO_PATH)
        image_lst = []
        iteration = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_lst.append(frame)
                print(f"Reading Frame #{iteration}", end="\r")
                iteration += 1
            else:
                break
        print("Complete. Saving...", end="")
        images_np = np.array(image_lst)
        np.save(absolute_images_np_path, images_np)
        print("Done.", end="\r")

    print("Loading NP Array...", end="")
    image_arr = np.load(absolute_images_np_path)
    print("Done.")

    return image_arr


def create_or_update_log(class_labels):
    if os.path.exists(OUTPUT_CSV_PATH):
        df = pd.read_csv(OUTPUT_CSV_PATH)
    else:
        df = pd.DataFrame()

    num_cols = len(df.columns)
    df[f"Run_{num_cols+1}"] = class_labels
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    return df


def create_modal_csv(df):
    # take modal class across columns in dataframe
    modal_series = df.mode(axis=1)[0]
    modal_series = modal_series.astype(int)

    modal_series.to_csv(
        os.path.join(OUTPUT_DIR, f"{video_name}_modal_labels.csv"), index=False
    )


def classify_images(im_arr):
    max_iteration = im_arr.shape[0]
    current_iteration, current_moment = 0, INITIAL_MOMENT

    # setup window
    window_name = "Classify frame"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 0, 0)
    print("press any key to begin")
    cv2.waitKey(0)

    class_labels = []

    while current_iteration < max_iteration:
        im_rgb = im_arr[current_iteration]
        # convert image to bgr from rgb
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

        coloured_image = cv2.applyColorMap(
            im_bgr,
            OPEN_CV_COLOUR_MAP[current_reps % len(OPEN_CV_COLOUR_MAP)],
        )
        cv2.imshow(window_name, coloured_image)

        key = cv2.waitKey(1000 // DEFAULT_PLAYBACK_FPS)
        if key == -1:
            # no input
            class_labels.append(current_moment)
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()
        elif key == ord("n"):
            current_moment = get_next_moment(current_moment)
            class_labels.append(current_moment)
        elif key == ord(" "):
            print()
            print("Paused. Press enter to continue.")
            cv2.waitKey(0)
            continue
        else:
            # any other key, repeat current iteration
            print(f"Invalid key pressed")
            continue

        current_iteration += 1

    print()
    print(f"Succesfully wrote {current_iteration+1} frames.")

    df = create_or_update_log(class_labels)
    create_modal_csv(df)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    output_iteration = 0

    print("Loading images to memory...", end="")
    np_im_arr = np_array_from_images()
    print("Done")

    classify_images(np_im_arr)
