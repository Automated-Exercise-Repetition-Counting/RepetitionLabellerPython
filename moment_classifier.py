import cv2
import os
import numpy as np
import moment_map

# User Input
ABSOLUTE_VIDEO_PATH = "G:\Shared drives\P4P\Data Collection\Squats\IMG_2167.MOV"
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT

# Constants
OUTPUT_DIR = "output"
INITIAL_MOMENT = 0
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT
DEFAULT_PLAYBACK_FPS = 10
OPEN_CV_COLOUR_MAP = [0, 1, 3, 5, 6, 7, 8, 10, 11]

video_name = os.path.basename(ABSOLUTE_VIDEO_PATH).split(".")[0]
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
                image_lst.append(frame)
                print(f"Reading Frame #{iteration}", end="\r")
                iteration += 1
            else:
                break
        print("Complete. Saving...", end="")
        images_np = np.array(image_lst)
        np.save(absolute_images_np_path, images_np)
        print("Done.", end="\r")

    return np.load(absolute_images_np_path)


def get_next_moment(current_moment):
    return (current_moment + 1) % len(MOMENT_MAP)


def update_frame_class(output_csv_path, frame_index, class_name):
    with open(output_csv_path, "a") as output_file:
        output_file.write(f"{frame_index},{class_name}\n")


def get_current_iteration_from_file(output_csv_path):
    if os.path.exists(output_csv_path):
        with open(output_csv_path, "r") as output_file:
            lines = output_file.readlines()
            if len(lines) > 0:
                iteration, class_name = lines[-1].split(",")
                return int(iteration), int(class_name)

    return 0, INITIAL_MOMENT


def classify_images(output_csv_path, im_arr):
    max_iteration = im_arr.shape[0]
    current_iteration, current_moment = get_current_iteration_from_file(output_csv_path)

    # setup window
    window_name = "Classify frame"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 0, 0)
    print("press any key to begin")
    cv2.waitKey(0)

    while current_iteration < max_iteration:
        print(f"\rCurrent Moment: {MOMENT_MAP[current_moment]}            ", end="")
        coloured_image = cv2.applyColorMap(
            im_arr[current_iteration],
            OPEN_CV_COLOUR_MAP[current_moment % len(OPEN_CV_COLOUR_MAP)],
        )
        cv2.imshow(window_name, coloured_image)

        match cv2.waitKey(1000 // DEFAULT_PLAYBACK_FPS):
            case -1:
                # no input
                update_frame_class(output_csv_path, current_iteration, current_moment)
            case 113:
                # q
                cv2.destroyAllWindows()
                exit()
            case 110:
                # n
                current_moment = get_next_moment(current_moment)
                update_frame_class(output_csv_path, current_iteration, current_moment)
            case 32:
                # space
                print()
                print("Paused. Press enter to continue.")
                cv2.waitKey(0)
            case _:
                # any other key, repeat current iteration
                print(f"Invalid key pressed")
                continue

        current_iteration += 1

    print()
    print(f"Succesfully wrote {current_iteration+1} frames.")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    output_iteration = 0

    try:
        output_iteration = int(
            input("Which output iteration to make/continue? (default 0) ")
        )
        if output_iteration < 0:
            raise ValueError
    except:
        output_iteration = 0

    output_csv_path = os.path.join(
        OUTPUT_DIR, f"{video_name}_output_{output_iteration}.csv"
    )

    print("Loading images to memory...", end="")
    np_im_arr = np_array_from_images()
    print("Done")

    classify_images(output_csv_path, np_im_arr)
