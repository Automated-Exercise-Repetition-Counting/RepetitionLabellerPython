import cv2
import os
import numpy as np
import pandas as pd


# User Input
ABSOLUTE_VIDEO_PATH = "G:\Shared drives\P4P\Data Collection\Squats\IMG_2167.MOV"
OUTPUT_DIR_BASEPATH = os.getcwd()
DEFAULT_PLAYBACK_FPS = 30

# Constants
OPEN_CV_COLOUR_MAP = [0, 1, 3, 5]

# Inferred Constants
video_name = os.path.basename(ABSOLUTE_VIDEO_PATH).split(".")[0]
output_dir = f"{video_name}_output"
output_csv_path = os.path.join(OUTPUT_DIR_BASEPATH, output_dir, f"{video_name}_labels.csv")
absolute_images_np_path = os.path.join(output_dir, f"{video_name}_images") + ".npy"


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
        images_np = np.array(image_lst)

        if images_np.shape[0] == 0:
            print("No images found. Exiting...")
            exit()

        print("Complete. Saving...", end="")
        np.save(absolute_images_np_path, images_np)
        print("Done.", end="\r")

    print("Loading NP Array...", end="")
    image_arr = np.load(absolute_images_np_path)
    print("Done.")

    return image_arr


def create_or_update_log(class_labels):
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
    else:
        df = pd.DataFrame()

    num_cols = len(df.columns)
    df[f"Run_{num_cols+1}"] = class_labels
    df.to_csv(output_csv_path, index=False)

    return df


def create_modal_csv(df):
    # take modal class across columns in dataframe
    modal_series = df.mode(axis=1)[0]
    modal_series = modal_series.astype(int)

    # rename to "Modal Rep"
    modal_series.rename("Modal Rep", inplace=True)
    modal_prefix = f"{video_name}_modal_labels_"

    num_runs = len(df.columns)
    if num_runs > 2:
        # delete previous modal csv if it exists

        # list files in ouput dir, and filter out non-csv files
        files = os.listdir(output_dir)
        csv_files = [f for f in files if modal_prefix in f]
        for f in csv_files:
            os.remove(os.path.join(output_dir, f))

        # save new csv
        modal_series.to_csv(
            os.path.join(output_dir, f"{modal_prefix}{num_runs}_runs.csv"), index=False
        )


def classify_images(im_arr):
    max_iteration = im_arr.shape[0]
    current_iteration = current_reps = 0

    # setup window
    window_name = "Classify frame"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 0, 0)
    print("press any key to begin")
    cv2.waitKey(0)

    class_labels = []

    while current_iteration < max_iteration:
        print(f"\rCurrent Reps: {current_reps}            ", end="")

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
            class_labels.append(current_reps)
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()
        elif key == ord("n"):
            current_reps += 1
            class_labels.append(current_reps)
        elif key == ord(" "):
            print()
            print("Paused. Press enter to continue.")
            cv2.waitKey(0)
            continue
        elif key == ord("r"):
            cv2.destroyAllWindows()
            return
        else:
            # any other key, repeat current iteration
            print(f"Invalid key pressed")
            continue

        current_iteration += 1

    cv2.destroyAllWindows()
    print()
    print(f"Succesfully wrote {current_iteration+1} frames.")
    df = create_or_update_log(class_labels)
    create_modal_csv(df)


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_iteration = 0

    print("Loading images to memory...", end="")
    np_im_arr = np_array_from_images()
    print("Done")

    not_done = True
    while not_done: 
        classify_images(np_im_arr)
        not_done = input("Classify the same video again? (y/[n]) ") == "y"

