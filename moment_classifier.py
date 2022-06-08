import cv2
import os
import numpy as np
import moment_map

# User Input
ABSOLUTE_VIDEO_PATH = "G:\Shared drives\P4P\Data Collection\Squats\IMG_2167.MOV"
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT

# Constants
OUTPUT_DIR = "output"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "output.csv")
INITIAL_MOMENT = 0
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
                print(f"\rReading Frame #{iteration}", end="")
                iteration += 1
            else:
                break
        print()
        print("Complete. Saving...", end="")
        images_np = np.array(image_lst)
        np.save(absolute_images_np_path, images_np)
        print("Done.")

    return np.load(absolute_images_np_path)


def get_frame_name_from_iteration(iteration):
    return video_name + "_frame_" + str(iteration) + ".jpg"


def get_frame_absolute_path(frame_name):
    return os.path.join(OUTPUT_DIR, frame_name)


def get_next_moment(current_moment):
    return (current_moment + 1) % len(MOMENT_MAP)


def update_frame_class(frame_index, class_name):
    with open(OUTPUT_CSV_PATH, "a") as output_file:
        output_file.write(f"{frame_index},{class_name}\n")


def remove_last_frame_update():
    with open(OUTPUT_CSV_PATH, "rw") as output_file:
        all_lines = output_file.readlines()
        output_file.write(all_lines[:-1])


def delete_frame(absolute_file_path):
    os.remove(absolute_file_path)


def get_current_iteration_from_file():
    with open(OUTPUT_CSV_PATH, "r") as output_file:
        lines = output_file.readlines()
        if len(lines) == 0:
            return 0
        else:
            return int(lines[-1].split(",")[0])


def classify_images(im_arr):
    max_iteration = im_arr.shape[0]
    current_iteration = get_current_iteration_from_file()
    current_moment = INITIAL_MOMENT

    while current_iteration <= max_iteration:
        frame_name = get_frame_name_from_iteration(current_iteration)
        frame_absolute_path = get_frame_absolute_path(frame_name)
        frame = cv2.imread(frame_absolute_path)

        if frame is None:
            current_iteration += 1
            continue

        window_name = "Classify " + frame_name

        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        match key:
            case 113 | -1:
                # q
                exit()
            case 110:
                # n
                update_frame_class(frame_name, current_moment)
            case 98:
                # b
                remove_last_frame_update()
                current_iteration -= 1
                continue
            case 32:
                # space
                current_moment = get_next_moment(current_moment)
                print(f"\rCurrent Moment: {current_moment}", end="")
                update_frame_class(frame_name, current_moment)
            case 100:
                # d
                # delete the current frame
                delete_frame(frame_absolute_path)
            case _:
                # any other key, repeat current iteration
                print(f"Invalid key pressed: {key}")
                continue

        current_iteration += 1

    print(f"Succesfully wrote {current_iteration+1} frames.")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    print("Loading images to memory...", end="")
    np_im_arr = np_array_from_images()
    print("Done loading images.")

    # classify_images(np_im_arr)
