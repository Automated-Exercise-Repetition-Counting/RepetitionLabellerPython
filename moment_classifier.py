import cv2
import os
import moment_map

# User Input
ABSOLUTE_VIDEO_PATH = "G:\Shared drives\P4P\Data Collection\Squats\IMG_2167.MOV"
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT

# Constants
OUTPUT_DIR = "output"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "output.csv")
INITIAL_MOMENT = 0
video_name = os.path.basename(ABSOLUTE_VIDEO_PATH).split(".")[0]


def write_all_images():
    cap = cv2.VideoCapture(ABSOLUTE_VIDEO_PATH)
    iteration = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(
                get_frame_absolute_path(get_frame_name_from_iteration(iteration)),
                frame,
            )
        else:
            break

        # write an updating frame number to the same terminal line
        print(f"\rWrote Frame {iteration}", end="")
        iteration += 1


def get_frame_name_from_iteration(iteration):
    return video_name + "_frame_" + str(iteration) + ".jpg"


def get_frame_absolute_path(frame_name):
    return os.path.join(OUTPUT_DIR, frame_name)


def get_next_moment(current_moment):
    return (current_moment + 1) % len(MOMENT_MAP)


def update_frame_class(frame_name, class_name):
    output_file = open(OUTPUT_CSV_PATH, "a")
    output_file.write(f"{frame_name},{class_name}\n")
    output_file.close()


def remove_last_frame_update():
    output_file = open(OUTPUT_CSV_PATH, "w")
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
            return int(lines[-1].split(",")[0].split("_")[-1])


def get_max_iteration_from_dir():
    img_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
    max_file_name = max(img_files, key=lambda f: int("".join(filter(str.isdigit, f))))
    return int("".join(filter(str.isdigit, max_file_name.split("_")[-1])))


def classify_images():
    max_iteration = get_max_iteration_from_dir()
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
        write_all_images()

    classify_images()
