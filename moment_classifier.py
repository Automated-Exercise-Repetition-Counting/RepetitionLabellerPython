import cv2
import os
import moment_map

# User Input
ABSOLUTE_VIDEO_PATH = ""
MOMENT_MAP = moment_map.MOMENT_MAP_SQUAT

# Constants
OUTPUT_DIR = "output"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "output.csv")
INITIAL_MOMENT = "0"
video_name = os.path.basename(ABSOLUTE_VIDEO_PATH)


def get_next_image():
    cap = cv2.VideoCapture(ABSOLUTE_VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            return None


def process_frame(frame, frame_name, current_moment):
    update_frame_class(frame_name, current_moment)
    # write image to output dir
    cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), frame)


def get_next_moment(current_moment):
    return (current_moment + 1) % len(MOMENT_MAP)


def update_frame_class(frame_name, class_name):
    output_file = open(OUTPUT_CSV_PATH, "w")
    output_file.write(frame_name + "," + class_name + "\n")
    output_file.close()


def remove_last_frame_update():
    output_file = open(OUTPUT_CSV_PATH, "w")
    all_lines = output_file.readlines()
    output_file.write(all_lines[:-1])


def classify_video():
    iteration = 0
    current_moment = INITIAL_MOMENT

    while True:
        frame = get_next_image()
        if frame is None:
            break

        frame_name = "frame_" + str(iteration) + ".jpg"

        cv2.imshow(MOMENT_MAP[current_moment], frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        match key:
            case "q":
                exit()
            case "39":
                # right arrow
                process_frame(frame, frame_name, current_moment)
            case "37":
                # left arrow
                remove_last_frame_update()
                iteration -= 1
                continue
            case "32":
                # space
                current_moment = get_next_moment(current_moment)
                process_frame(frame, frame_name, current_moment)
            case _:
                # any other key, repeat current iteration
                print("Invalid key pressed: " + key)
                continue

        iteration += 1

    print(f"Succesfully wrote {iteration+1} frames.")


if __name__ == "__main__":
    classify_video()
